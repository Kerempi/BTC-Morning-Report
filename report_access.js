(function () {
  const storageKey = "report_access_2026_05_08_v2";
  const allowed = [
    "a544e08fc61fb752b842f09acaff1a801f6aec14e3b8a187184d1ca37db9d33f"
  ];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
