(function () {
  const storageKey = "report_access_2026_07_13_v1";
  const allowed = [
    "320f8da18c9bff2b3776f9aa2bc72eaee31fccd05b7683d6929a7ca8a916c586"
  ];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
