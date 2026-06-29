(function () {
  const storageKey = "report_access_2026_06_29_v1";
  const allowed = [
    "5eb2b0e3ae2604d7e041524fd20db0a2cadba30131591644443809b3a7c316ae"
  ];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
