(function () {
  const allowed = [
    "51216a202487373730d19aad238f481ab4ee8895a0e13297233308dccafcda88"
  ];
  const saved = sessionStorage.getItem("report_access");
  if (!saved || !allowed.includes(saved)) {
    window.location.replace("index.html");
  }
})();
