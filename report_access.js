(function () {
  const allowed = [
    "7f36521c46fe193a1a19bd16e77e4ddb8191eee92ed9a53a359d8ac4979c6541"
  ];
  const saved = sessionStorage.getItem("report_access");
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
