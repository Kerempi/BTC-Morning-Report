(function () {
  const allowed = [
    "a6ea19515d26eb9273a4e4baf13a14430cc0e9387711ae44f0cc2b53a20e8bc9"
  ];
  const saved = sessionStorage.getItem("report_access");
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
