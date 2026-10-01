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

  // Point every [data-download] button at the visitor's platform and mark the
  // matching card in the downloads section as recommended.
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

  var header = document.querySelector(".site-header");
  function onScroll() {
    if (header) header.classList.toggle("scrolled", window.scrollY > 8);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if ("IntersectionObserver" in window) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px" }
    );
    document.querySelectorAll(".reveal").forEach(function (el) {
      observer.observe(el);
    });
  } else {
    document.querySelectorAll(".reveal").forEach(function (el) {
      el.classList.add("visible");
    });
  }
})();
