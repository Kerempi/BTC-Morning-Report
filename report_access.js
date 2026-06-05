(function () {
  const storageKey = "report_access_2026_06_05_v1";
  const allowed = [
    "24568073fafeea17a4ea4421c33c97f68c57d9b24a8887d4eb19b1de5510b2a5"
  ];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
