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

  // The hero demo: a display-arrangement panel whose selection follows a
  // simulated gaze. Display 2 holds a window split in two panes, so the loop
  // also shows split-pane focus: the focus moves to the looked-at pane while
  // the cursor stays put. The visitor can take over by pointing at or clicking
  // a display or a pane. Nothing here uses the camera.
  function arrange(figure) {
    var canvas = figure.querySelector(".arrange-canvas");
    var tiles = Array.prototype.slice.call(figure.querySelectorAll(".tile"));
    var panes = Array.prototype.slice.call(figure.querySelectorAll("[data-pane]"));
    var paneTile = panes.length ? panes[0].closest(".tile") : null;
    var pointer = figure.querySelector(".pointer");
    var ray = figure.querySelector(".ray");
    var cam = figure.querySelector(".cam");
    var desk = figure.querySelector(".desk");
    var status = figure.querySelector("[data-status]");
    var identify = figure.querySelector("[data-identify]");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    // Where the cursor was last left on each display (fractions of the tile).
    var lastSpot = { 1: [0.66, 0.6], 2: [0.3, 0.55] };
    // Look at display 1, glance at the phone (ignored), look back at display 2,
    // then at its other pane.
    var script = panes.length > 1 ? ["1", "away", "2", "pane"] : ["1", "away", "2"];
    var focusedPane = 0;
    var step = 0;
    var current = "2";
    var target = "2";
    var timer = null;
    var resumeAt = 0;
    var visible = false;

    function message(name, display) {
      var text = figure.getAttribute("data-msg-" + name) || "";
      var pane = panes[focusedPane];
      return text
        .replace("{n}", display)
        .replace("{pane}", (pane && pane.getAttribute("data-label")) || "");
    }

    function centre(el) {
      var box = el.getBoundingClientRect();
      var origin = canvas.getBoundingClientRect();
      return {
        x: box.left - origin.left + box.width / 2,
        y: box.top - origin.top + box.height / 2,
        box: box,
        origin: origin,
      };
    }

    function layout() {
      var tile = tiles.filter(function (t) {
        return t.getAttribute("data-display") === current;
      })[0];
      if (!tile) return;
      var t = centre(tile);
      var spot = lastSpot[current];
      var px = t.box.left - t.origin.left + t.box.width * spot[0];
      var py = t.box.top - t.origin.top + t.box.height * spot[1];
      pointer.style.transform = "translate(" + px.toFixed(1) + "px," + py.toFixed(1) + "px)";

      // The ray starts under the webcam and runs behind the displays, so only
      // the stretch between the camera and the looked-at target shows.
      var from = centre(cam);
      var fromY = from.y + from.box.height / 2;
      var to = t;
      if (target === "away") to = centre(desk);
      else if (target === "pane" && panes[focusedPane]) to = centre(panes[focusedPane]);
      var dx = to.x - from.x;
      var dy = to.y - fromY;
      ray.style.left = from.x.toFixed(1) + "px";
      ray.style.top = fromY.toFixed(1) + "px";
      ray.style.width = Math.sqrt(dx * dx + dy * dy).toFixed(1) + "px";
      ray.style.transform = "rotate(" + Math.atan2(dy, dx).toFixed(4) + "rad)";
    }

    function showPane(index) {
      focusedPane = index;
      panes.forEach(function (pane, i) {
        pane.classList.toggle("is-focus", i === index);
      });
    }

    // Split-pane focus: only on the display already looked at; the cursor
    // does not move.
    function selectPane(index, reason) {
      var display = paneTile ? paneTile.getAttribute("data-display") : null;
      if (!display) return;
      if (current !== display) select(display, reason);
      showPane(index);
      target = "pane";
      figure.classList.remove("away");
      if (status) status.textContent = message("pane", display);
      layout();
    }

    function select(display, reason) {
      if (display === "pane") {
        selectPane(panes.length > 1 ? 1 - focusedPane : 0, reason);
        return;
      }
      target = display;
      figure.classList.toggle("away", display === "away");
      if (display !== "away") {
        current = display;
        tiles.forEach(function (tile) {
          tile.setAttribute(
            "aria-pressed",
            tile.getAttribute("data-display") === display ? "true" : "false"
          );
        });
      }
      if (status) {
        status.textContent =
          display === "away" ? message("away") : message(reason || "look", display);
      }
      layout();
    }

    function tick() {
      if (Date.now() < resumeAt) return;
      select(script[step % script.length]);
      step += 1;
    }

    function start() {
      if (timer || reduceMotion.matches || !visible || document.hidden) return;
      timer = window.setInterval(tick, 2800);
    }

    function stop() {
      window.clearInterval(timer);
      timer = null;
    }

    function takeOver(tile) {
      resumeAt = Date.now() + 8000;
      select(tile.getAttribute("data-display"), "pick");
    }

    function takeOverPane(pane) {
      resumeAt = Date.now() + 8000;
      var index = panes.indexOf(pane);
      if (target === "pane" && index === focusedPane) return;
      selectPane(index, "pick");
    }

    tiles.forEach(function (tile) {
      tile.addEventListener("click", function (event) {
        var pane = event.target.closest && event.target.closest("[data-pane]");
        if (pane) takeOverPane(pane);
        else takeOver(tile);
      });
      tile.addEventListener("pointerenter", function (event) {
        if (event.pointerType === "mouse") takeOver(tile);
      });
    });
    panes.forEach(function (pane) {
      pane.addEventListener("pointerenter", function (event) {
        if (event.pointerType === "mouse") takeOverPane(pane);
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
    layout();
    window.requestAnimationFrame(function () {
      figure.classList.add("ready");
    });
  }
})();
