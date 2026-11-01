function skipSdkReplay(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:[-+].+)?$/.exec(version);
  if (!match) throw new Error(`Invalid onnx-light version: ${version}`);
  const [major, minor, patch] = match.slice(1).map(Number);
  return major === 0 && (minor < 2 || (minor === 2 && patch <= 3))
    ? `onnx-light ${version}: optimization replay blocked by xadupre/onnx-light#5756 (SDK <= 0.2.3).`
    : false;
}

module.exports = { skipSdkReplay };
