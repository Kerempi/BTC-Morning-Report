(function () {
  const storageKey = "report_access_2026_09_28_v3";
  const allowed = [
    "0d08e0bb3ecfd5bef6ec28d1420eec213ca385ff33d092d3cbba58220de6bbb3"
];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
