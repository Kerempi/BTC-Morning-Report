(function () {
  const storageKey = "report_access_2026_09_28_v2";
  const allowed = [
    "d562de096d354a8b8e4b8cab80cb9d1b53f7eaf41b55f6aa7dca28c1a24734ad"
];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
