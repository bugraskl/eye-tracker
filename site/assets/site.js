// Eye Tracker website. Progressive enhancement only: without JavaScript every
// link still works and the download button points to the downloads section.
// No analytics, no cookies, no storage, no requests to other sites.
(function () {
  "use strict";

  function detectPlatform() {
    var uaData = navigator.userAgentData;
    var platform = ((uaData && uaData.platform) || navigator.platform || "").toLowerCase();
    var ua = (navigator.userAgent || "").toLowerCase();
    if (/android|iphone|ipad|ipod/.test(ua)) return "mobile";
    if (platform.indexOf("win") !== -1 || ua.indexOf("windows") !== -1) return "windows";
    if (platform.indexOf("mac") !== -1 || ua.indexOf("mac os") !== -1) return "macos";
    if (platform.indexOf("linux") !== -1 || ua.indexOf("linux") !== -1) return "linux";
    return "";
  }

  // Point every [data-download] button at the visitor's platform and select
  // the matching tile in the downloads section.
  var platform = detectPlatform();
  document.querySelectorAll("[data-download]").forEach(function (button) {
    var target = platform && button.getAttribute("data-" + platform + "-href");
    var label = platform && button.getAttribute("data-" + platform + "-label");
    var meta = platform && button.getAttribute("data-" + platform + "-meta");
    if (!target || !label) return;
    button.setAttribute("href", target);
    var text = button.querySelector(".btn-label");
    if (text) text.textContent = label;
    var small = button.querySelector(".btn-meta");
    if (small) small.textContent = meta || "";
  });
  var card = platform && document.getElementById("download-" + platform);
  if (card) card.classList.add("recommended");

  // Show hotkeys the way the visitor's keyboard labels them.
  if (platform === "macos" || platform === "linux") {
    document.querySelectorAll("[data-keys-" + platform + "]").forEach(function (key) {
      key.textContent = key.getAttribute("data-keys-" + platform);
    });
  }

  var header = document.querySelector(".topbar");
  function onScroll() {
    if (header) header.classList.toggle("scrolled", window.scrollY > 8);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  document.querySelectorAll("[data-arrange]").forEach(arrange);


  // The hero demo: a display-arrangement panel driven by a simulated gaze.
  // A pair of eyes at the bottom (where you sit) turns towards what it looks
  // at, a gaze dot fills up while the glance dwells, and only then does the
  // selection move: to a display (cursor and keyboard focus follow), to the
  // other pane of the split window on display 2 (the same, inside the
  // window), or nowhere for a glance at the phone. The visitor can take over by
  // pointing at or clicking a display or a pane. Nothing here uses the camera.
  function arrange(figure) {
    var canvas = figure.querySelector(".arrange-canvas");
    var tiles = Array.prototype.slice.call(figure.querySelectorAll(".tile"));
    var panes = Array.prototype.slice.call(figure.querySelectorAll("[data-pane]"));
    var paneTile = panes.length ? panes[0].closest(".tile") : null;
    var pointer = figure.querySelector(".pointer");
    var ray = figure.querySelector(".ray");
    var desk = figure.querySelector(".desk");
    var eyes = figure.querySelector(".eyes");
    var irises = Array.prototype.slice.call(figure.querySelectorAll(".eye i"));
    var dot = figure.querySelector(".gaze-dot");
    var status = figure.querySelector("[data-status]");
    var identify = figure.querySelector("[data-identify]");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    var DWELL_MS = 700;
    var PICK_DWELL_MS = 250;
    var STEP_MS = 3000;
    // Where the cursor was last left: on display 1 (fractions of the tile) and
    // in each pane of display 2 (fractions of the pane).
    var displaySpot = { 1: [0.66, 0.6] };
    var paneSpot = [[0.5, 0.62], [0.42, 0.5]];
    // Look at display 1, glance at the phone (ignored), look back at display 2,
    // then at its other pane.
    var script = panes.length > 1 ? ["1", "away", "2", "pane"] : ["1", "away", "2"];
    var step = 0;
    var current = paneTile ? paneTile.getAttribute("data-display") : "2";
    var focusedPane = 0;
    var aim = { kind: "pane", pane: 0 };
    var timer = null;
    var commitTimer = null;
    var resumeAt = 0;
    var visible = false;

    function message(name, display) {
      var text = figure.getAttribute("data-msg-" + name) || "";
      var pane = panes[focusedPane];
      return text
        .replace("{n}", display)
        .replace("{pane}", (pane && pane.getAttribute("data-label")) || "");
    }

    function box(el) {
      var b = el.getBoundingClientRect();
      var o = canvas.getBoundingClientRect();
      return { x: b.left - o.left, y: b.top - o.top, w: b.width, h: b.height };
    }

    function centre(el) {
      var b = box(el);
      return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
    }

    function tileOf(display) {
      return tiles.filter(function (t) {
        return t.getAttribute("data-display") === display;
      })[0];
    }

    function split(display) {
      return paneTile && paneTile.getAttribute("data-display") === display;
    }

    // The point the eyes look at for an aim.
    function gazePoint(a) {
      if (a.kind === "away") return centre(desk);
      if (a.kind === "pane") return centre(panes[a.pane]);
      if (split(a.display)) return centre(panes[focusedPane]);
      var tile = tileOf(a.display);
      return centre(tile.querySelector(".win") || tile);
    }

    // Where the cursor sits on the current display.
    function cursorPoint() {
      if (split(current)) {
        var p = box(panes[focusedPane]);
        var s = paneSpot[focusedPane];
        return { x: p.x + p.w * s[0], y: p.y + p.h * s[1] };
      }
      var t = box(tileOf(current));
      var d = displaySpot[current] || [0.5, 0.5];
      return { x: t.x + t.w * d[0], y: t.y + t.h * d[1] };
    }

    function place(el, point) {
      el.style.transform = "translate(" + point.x.toFixed(1) + "px," + point.y.toFixed(1) + "px)";
    }

    function layout() {
      place(pointer, cursorPoint());
      var target = gazePoint(aim);
      place(dot, target);

      // The gaze ray runs from between the eyes to the looked-at point.
      var e = box(eyes);
      var from = { x: e.x + e.w / 2, y: e.y + 6 };
      var dx = target.x - from.x;
      var dy = target.y - from.y;
      var length = Math.sqrt(dx * dx + dy * dy);
      ray.style.left = from.x.toFixed(1) + "px";
      ray.style.top = from.y.toFixed(1) + "px";
      ray.style.width = length.toFixed(1) + "px";
      ray.style.transform = "rotate(" + Math.atan2(dy, dx).toFixed(4) + "rad)";

      // Irises turn towards the point.
      var ix = length ? (dx / length) * 6 : 0;
      var iy = length ? (dy / length) * 3.5 : 0;
      irises.forEach(function (iris) {
        iris.style.transform = "translate(" + ix.toFixed(1) + "px," + iy.toFixed(1) + "px)";
      });
    }

    // Start a glance: eyes, ray and gaze dot move; the dot fills while it dwells.
    function look(a, dwell) {
      aim = a;
      figure.classList.toggle("away", a.kind === "away");
      dot.style.setProperty("--dwell", dwell + "ms");
      dot.classList.remove("dwelling", "settled");
      void dot.getBoundingClientRect(); // restart the fill animation
      dot.classList.add("dwelling");
      layout();
    }

    function showPane(index) {
      focusedPane = index;
      panes.forEach(function (pane, i) {
        pane.classList.toggle("is-focus", i === index);
      });
    }

    function selectDisplay(display) {
      current = display;
      tiles.forEach(function (tile) {
        tile.setAttribute("aria-pressed", tile.getAttribute("data-display") === display ? "true" : "false");
      });
    }

    // The dwell is over: move the selection (or ignore the glance).
    function commit(a, reason) {
      dot.classList.remove("dwelling");
      dot.classList.add("settled");
      if (a.kind === "away") {
        if (status) status.textContent = message("away");
      } else if (a.kind === "pane") {
        var display = paneTile.getAttribute("data-display");
        if (current !== display) selectDisplay(display);
        showPane(a.pane);
        if (status) status.textContent = message("pane", display);
      } else {
        selectDisplay(a.display);
        if (status) status.textContent = message(reason || "look", a.display);
      }
      layout();
    }

    function glance(a, dwell, reason) {
      window.clearTimeout(commitTimer);
      look(a, dwell);
      if (reduceMotion.matches) {
        commit(a, reason);
        return;
      }
      commitTimer = window.setTimeout(function () {
        commit(a, reason);
      }, dwell);
    }

    function aimFor(name) {
      if (name === "away") return { kind: "away" };
      if (name === "pane") {
        // Only a pane of the display already looked at.
        var other = panes.length > 1 ? 1 - focusedPane : 0;
        return { kind: "pane", pane: other };
      }
      return { kind: "display", display: name };
    }

    function tick() {
      if (Date.now() < resumeAt) return;
      glance(aimFor(script[step % script.length]), DWELL_MS);
      step += 1;
    }

    function start() {
      if (timer || reduceMotion.matches || !visible || document.hidden) return;
      timer = window.setInterval(tick, STEP_MS);
    }

    function stop() {
      window.clearInterval(timer);
      timer = null;
    }

    function takeOver(a) {
      resumeAt = Date.now() + 8000;
      if (aim.kind === a.kind && aim.display === a.display && aim.pane === a.pane) return;
      glance(a, PICK_DWELL_MS, "pick");
    }

    tiles.forEach(function (tile) {
      var display = tile.getAttribute("data-display");
      tile.addEventListener("click", function (event) {
        var pane = event.target.closest && event.target.closest("[data-pane]");
        if (pane) takeOver({ kind: "pane", pane: panes.indexOf(pane) });
        else takeOver({ kind: "display", display: display });
      });
      tile.addEventListener("pointerenter", function (event) {
        if (event.pointerType === "mouse" && !split(display)) {
          takeOver({ kind: "display", display: display });
        }
      });
    });
    panes.forEach(function (pane, index) {
      pane.addEventListener("pointerenter", function (event) {
        if (event.pointerType === "mouse") takeOver({ kind: "pane", pane: index });
      });
    });

    if (identify) {
      identify.addEventListener("click", function () {
        figure.classList.add("identifying");
        window.setTimeout(function () {
          figure.classList.remove("identifying");
        }, 2200);
      });
    }

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible) start();
        else stop();
      }).observe(figure);
    } else {
      visible = true;
      start();
    }
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) stop();
      else start();
    });
    if (reduceMotion.addEventListener) {
      reduceMotion.addEventListener("change", function () {
        if (reduceMotion.matches) stop();
        else start();
      });
    }
    window.addEventListener("resize", layout);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);
    dot.classList.add("settled");
    layout();
    window.requestAnimationFrame(function () {
      figure.classList.add("ready");
    });
  }
})();
