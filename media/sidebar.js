function createSidebar({ byId, fit, endGesture, onMetadataToggle }) {
  function revealSidebarPanel(panel) {
    panel.parentElement.scrollTop = Math.max(
      0,
      panel.offsetTop - byId("sidebar-controls").offsetHeight - 12,
    );
  }
  for (const id of ["show-operator", "show-find", "show-metadata"])
    byId(id).addEventListener("click", () => {
      const button = byId(id);
      const panel = byId(button.getAttribute("aria-controls"));
      panel.hidden = !panel.hidden;
      button.setAttribute("aria-expanded", String(!panel.hidden));
      if (!panel.hidden) revealSidebarPanel(panel);
      if (id === "show-metadata") onMetadataToggle(!panel.hidden);
    });
  const sidebar = byId("graph-panels");
  const sidebarResizer = byId("sidebar-resizer");
  const graphBody = sidebar.parentElement;
  const sidebarStyle = getComputedStyle(graphBody);
  const sidebarMinimumWidth = parseFloat(
    sidebarStyle.getPropertyValue("--sidebar-min-width"),
  );
  const graphMinimumWidth = parseFloat(
    sidebarStyle.getPropertyValue("--graph-min-width"),
  );
  const sidebarResizerWidth = parseFloat(
    sidebarStyle.getPropertyValue("--sidebar-resizer-width"),
  );
  const sidebarKeyboardStep = 20;
  let sidebarDrag;
  let sidebarFitPending = false;

  function maximumSidebarWidth() {
    return Math.max(
      sidebarMinimumWidth,
      graphBody.getBoundingClientRect().width -
        graphMinimumWidth -
        sidebarResizerWidth,
    );
  }

  function updateSidebarSize() {
    const width = Math.round(sidebar.getBoundingClientRect().width);
    if (!width) return;
    sidebarResizer.setAttribute("aria-valuemin", String(sidebarMinimumWidth));
    sidebarResizer.setAttribute("aria-valuemax", String(maximumSidebarWidth()));
    sidebarResizer.setAttribute("aria-valuenow", String(width));
    sidebarResizer.setAttribute("aria-valuetext", `${width} pixels`);
  }

  function scheduleSidebarFit() {
    if (sidebarFitPending) return;
    sidebarFitPending = true;
    window.requestAnimationFrame(() => {
      sidebarFitPending = false;
      fit();
    });
  }

  function resizeSidebar(width) {
    graphBody.style.setProperty(
      "--sidebar-width",
      `${Math.round(Math.max(sidebarMinimumWidth, Math.min(maximumSidebarWidth(), width)))}px`,
    );
    updateSidebarSize();
    scheduleSidebarFit();
  }

  function endSidebarDrag(event) {
    if (sidebarDrag?.pointerId !== event.pointerId) return;
    sidebarDrag = undefined;
    document.body.classList.remove("resizing-sidebar");
    if (sidebarResizer.hasPointerCapture(event.pointerId)) {
      sidebarResizer.releasePointerCapture(event.pointerId);
    }
  }

  sidebarResizer.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || sidebarDrag) return;
    event.preventDefault();
    endGesture();
    sidebarDrag = {
      pointerId: event.pointerId,
      x: event.clientX,
      width: sidebar.getBoundingClientRect().width,
    };
    sidebarResizer.setPointerCapture(event.pointerId);
    sidebarResizer.focus();
    document.body.classList.add("resizing-sidebar");
  });
  sidebarResizer.addEventListener("pointermove", (event) => {
    if (sidebarDrag?.pointerId !== event.pointerId) return;
    resizeSidebar(sidebarDrag.width + sidebarDrag.x - event.clientX);
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    sidebarResizer.addEventListener(type, endSidebarDrag);
  }
  sidebarResizer.addEventListener("keydown", (event) => {
    const width = sidebar.getBoundingClientRect().width;
    const widths = {
      ArrowLeft: width + sidebarKeyboardStep,
      ArrowRight: width - sidebarKeyboardStep,
      Home: sidebarMinimumWidth,
      End: graphBody.getBoundingClientRect().width,
    };
    if (!Object.prototype.hasOwnProperty.call(widths, event.key)) return;
    event.preventDefault();
    endGesture();
    resizeSidebar(widths[event.key]);
  });
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      updateSidebarSize();
      scheduleSidebarFit();
    }).observe(sidebar);
  }
  updateSidebarSize();
  window.addEventListener("resize", () => {
    if (sidebarDrag) endSidebarDrag(sidebarDrag);
    endGesture();
    updateSidebarSize();
    fit();
  });

  return { revealPanel: revealSidebarPanel };
}
