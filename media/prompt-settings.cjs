function genaiSessionOptions(settings, profilePath) {
  const options = settings.sessionOptions
    ? JSON.parse(settings.sessionOptions)
    : {};
  for (const [key, name] of [
    ["intraOpThreads", "intra_op_num_threads"],
    ["interOpThreads", "inter_op_num_threads"],
    ["graphOptimization", "graph_optimization_level"],
    ["cpuArena", "enable_cpu_mem_arena"],
    ["memoryPattern", "enable_mem_pattern"],
    ["logSeverity", "log_severity_level"],
    ["logVerbosity", "log_verbosity_level"],
    ["logId", "log_id"],
    ["customOpsLibrary", "custom_ops_library"],
  ])
    if (settings[key] !== undefined) options[name] = settings[key];
  if (settings.parallelExecution !== undefined)
    options["session.execution_mode"] = settings.parallelExecution
      ? "parallel"
      : "sequential";
  if (settings.prepacking !== undefined)
    options["session.disable_prepacking"] = settings.prepacking ? "0" : "1";
  if (settings.profile) options.enable_profiling = profilePath;

  const selected =
    settings.providers ??
    (settings.provider === undefined ? undefined : [settings.provider]);
  if (selected) {
    if (selected.includes("cpu"))
      options["session.disable_cpu_ep_fallback"] = "0";
    const previous = Array.isArray(options.provider_options)
      ? options.provider_options
      : [];
    // Remove deselected/unknown JSON providers before the SDK parses the overlay.
    options.provider_options = previous.flatMap((entry) =>
      entry && typeof entry === "object" && !Array.isArray(entry)
        ? Object.entries(entry).flatMap(([name, value]) => {
            const provider = name.toLowerCase();
            return selected.includes(provider) ? [{ [name]: value }] : [];
          })
        : [],
    );
  }
  const providers =
    selected ??
    (Array.isArray(options.provider_options)
      ? [
          ...new Set(
            options.provider_options.flatMap((entry) =>
              entry && typeof entry === "object" && !Array.isArray(entry)
                ? Object.keys(entry)
                : [],
            ),
          ),
        ]
      : undefined);
  return { options, providers };
}

module.exports = { genaiSessionOptions };
