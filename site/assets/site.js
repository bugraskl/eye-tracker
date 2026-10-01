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
  // simulated gaze. The visitor can take over by pointing at or clicking a
  // display. Nothing here uses the camera.
  function arrange(figure) {
    var canvas = figure.querySelector(".arrange-canvas");
    var tiles = Array.prototype.slice.call(figure.querySelectorAll(".tile"));
    var pointer = figure.querySelector(".pointer");
    var ray = figure.querySelector(".ray");
    var cam = figure.querySelector(".cam");
    var desk = figure.querySelector(".desk");
    var status = figure.querySelector("[data-status]");
    var identify = figure.querySelector("[data-identify]");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    // Where the cursor was last left on each display (fractions of the tile).
    var lastSpot = { 1: [0.66, 0.6], 2: [0.42, 0.5] };
    var script = ["1", "away", "2"];
    var step = 0;
    var current = "2";
    var target = "2";
    var timer = null;
    var resumeAt = 0;
    var visible = false;

    function message(name, display) {
      var text = figure.getAttribute("data-msg-" + name) || "";
      return text.replace("{n}", display);
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
      var to = target === "away" ? centre(desk) : t;
      var dx = to.x - from.x;
      var dy = to.y - fromY;
      ray.style.left = from.x.toFixed(1) + "px";
      ray.style.top = fromY.toFixed(1) + "px";
      ray.style.width = Math.sqrt(dx * dx + dy * dy).toFixed(1) + "px";
      ray.style.transform = "rotate(" + Math.atan2(dy, dx).toFixed(4) + "rad)";
    }

    function select(display, reason) {
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

    tiles.forEach(function (tile) {
      tile.addEventListener("click", function () {
        takeOver(tile);
      });
      tile.addEventListener("pointerenter", function (event) {
        if (event.pointerType === "mouse") takeOver(tile);
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
