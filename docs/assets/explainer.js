/* Shared scrollytelling helpers for the explainers.
   Every page: one SVG chart driven through states by IntersectionObserver.
   Requires d3 (vendored) loaded first. */

window.LabExplainer = (function () {
  const css = getComputedStyle(document.documentElement);
  const C = {
    naive: css.getPropertyValue("--naive").trim(),
    corrected: css.getPropertyValue("--corrected").trim(),
    secondary: css.getPropertyValue("--secondary").trim(),
    tertiary: "#eda100",
    muted: css.getPropertyValue("--ink-muted").trim(),
    grid: css.getPropertyValue("--grid").trim(),
    ink: css.getPropertyValue("--ink-secondary").trim(),
    baseline: css.getPropertyValue("--baseline").trim(),
  };

  function chart(selector, opts) {
    const W = 420, H = 320;
    const M = Object.assign({ top: 18, right: 52, bottom: 40, left: 48 }, opts.margin);
    const svg = d3.select(selector);
    const gGrid = svg.append("g");
    const gx = svg.append("g").attr("transform", `translate(0,${H - M.bottom})`);
    const gy = svg.append("g").attr("transform", `translate(${M.left},0)`);
    const xTitle = svg.append("text")
      .attr("x", (M.left + W - M.right) / 2).attr("y", H - 6)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", C.muted);
    const yTitle = svg.append("text")
      .attr("transform", `translate(12,${(M.top + H - M.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", C.muted);

    const api = { svg, W, H, M, C };

    api.xLinear = (domain) => d3.scaleLinear().domain(domain).range([M.left, W - M.right]);
    api.xLog = (domain) => d3.scaleLog().domain(domain).range([M.left, W - M.right]);
    api.xBand = (names) => d3.scaleBand().domain(names)
      .range([M.left, W - M.right]).padding(0.35);
    api.y = (domain) => d3.scaleLinear().domain(domain).range([H - M.bottom, M.top]);

    api.axes = function (x, y, { xLabel = "", yLabel = "", yFmt = null, xTicks = 6, xFmt = null } = {}) {
      const bottom = d3.axisBottom(x).tickSize(0).tickPadding(8);
      if (x.bandwidth === undefined) bottom.ticks(xTicks, xFmt || undefined);
      gx.call(bottom);
      gy.call(d3.axisLeft(y).ticks(6).tickFormat(yFmt || undefined).tickSize(0).tickPadding(6));
      [gx, gy].forEach(g => {
        g.selectAll(".domain").attr("stroke", C.baseline);
        g.selectAll("text").attr("fill", C.muted);
      });
      gGrid.selectAll("line").data(y.ticks(6)).join("line")
        .attr("x1", M.left).attr("x2", W - M.right)
        .attr("y1", d => y(d)).attr("y2", d => y(d))
        .attr("stroke", C.grid).attr("stroke-width", 0.7);
      gGrid.lower();
      xTitle.text(xLabel);
      yTitle.text(yLabel);
    };

    api.truthLine = function (y, value, label, { color = C.muted, side = "left" } = {}) {
      const g = svg.append("g");
      g.append("line")
        .attr("x1", M.left).attr("x2", W - M.right)
        .attr("y1", y(value)).attr("y2", y(value))
        .attr("stroke", color).attr("stroke-dasharray", "5 4").attr("stroke-width", 1.5);
      g.append("text")
        .attr("x", side === "left" ? M.left + 4 : W - M.right - 4)
        .attr("y", y(value) - 6)
        .attr("text-anchor", side === "left" ? "start" : "end")
        .attr("font-size", 11).attr("fill", C.ink).text(label);
      return g;
    };

    // Bars that appear per state. items: [{name, value, color}]
    api.bars = function (x, y, items) {
      const g = svg.append("g");
      const sel = g.selectAll("g.bar").data(items, d => d.name).join(enter => {
        const b = enter.append("g").attr("class", "bar").attr("opacity", 0);
        b.append("rect")
          .attr("x", d => x(d.name)).attr("width", x.bandwidth())
          .attr("y", d => y(d.value))
          .attr("height", d => y(0) - y(d.value))
          .attr("rx", 3).attr("fill", d => d.color);
        b.append("text")
          .attr("x", d => x(d.name) + x.bandwidth() / 2).attr("y", d => y(d.value) - 6)
          .attr("text-anchor", "middle").attr("font-size", 12)
          .attr("font-weight", 600).attr("fill", d => d.color)
          .text(d => d.label !== undefined ? d.label : d.value);
        return b;
      });
      return {
        show(names, dur = 500) {
          sel.transition().duration(dur)
            .attr("opacity", d => names.includes(d.name) ? 1 : 0);
        },
        group: g,
      };
    };

    api.animatedLine = function (x, y, xs, ys, color) {
      const path = svg.append("path")
        .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.5)
        .attr("d", d3.line().x((d, i) => x(xs[i])).y(d => y(d))(ys));
      const len = path.node().getTotalLength();
      path.attr("stroke-dasharray", len).attr("stroke-dashoffset", len);
      return {
        path,
        draw(dur = 1400) {
          path.transition().duration(dur).ease(d3.easeCubicOut).attr("stroke-dashoffset", 0);
        },
        hide() { path.transition().duration(300).attr("stroke-dashoffset", len); },
        show() { path.attr("stroke-dashoffset", 0); },
      };
    };

    api.endLabel = function (x, y, xv, yv, text, color) {
      return svg.append("text")
        .attr("x", x(xv) + 5).attr("y", y(yv) + 4)
        .attr("font-size", 12).attr("font-weight", 600).attr("fill", color)
        .attr("opacity", 0).text(text);
    };

    return api;
  }

  function scroller(states, { caption = null } = {}) {
    const steps = document.querySelectorAll(".step");
    const cap = caption ? document.getElementById(caption) : null;
    const run = i => {
      const s = states[i];
      if (!s) return;
      s.enter();
      if (cap && s.caption) cap.textContent = s.caption;
    };
    run(0);
    const observer = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          steps.forEach(s => s.classList.remove("active"));
          e.target.classList.add("active");
          run(+e.target.dataset.step);
        }
      }
    }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach(s => observer.observe(s));
  }

  return { chart, scroller, C };
})();
