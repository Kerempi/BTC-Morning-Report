(function () {
  const storageKey = "report_access_2026_08_08_v1";
  const allowed = [
    "7bf273a1bab6097b37ed60eb25479ab749028cdcb538982e8f79fd1e380b871b"
];
  const saved = sessionStorage.getItem(storageKey);
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
