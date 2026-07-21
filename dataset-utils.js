const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let value = n;
    for (let bit = 0; bit < 8; bit += 1) value = (value & 1) ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    table[n] = value >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

export function csvText(lines) {
  return lines.map(row => row.map(value => `"${String(value ?? '').replace(/"/g, '""')}"`).join(',')).join('\r\n');
}

export function cocoBoxFromNormalized(box, imageWidth, imageHeight) {
  return [box.x * imageWidth, box.y * imageHeight, box.w * imageWidth, box.h * imageHeight].map(value => Number(value.toFixed(2)));
}

export function yoloLineFromNormalized(classId, box) {
  return `${classId} ${(box.x + box.w / 2).toFixed(6)} ${(box.y + box.h / 2).toFixed(6)} ${box.w.toFixed(6)} ${box.h.toFixed(6)}`;
}

function normaliseTaxonomyValue(value) {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}

export function taxonomyAliases(species = {}) {
  return [
    ['scientific', species.scientific],
    ['code', species.code],
    ['common', species.common]
  ].map(([kind, value]) => {
    const normalised = normaliseTaxonomyValue(value);
    return normalised ? `${kind}:${normalised}` : null;
  }).filter(Boolean);
}

export function mapLearningExamples(examples, species, options = {}) {
  const featureVersion = options.featureVersion ?? 1, excludeAnnotationId = options.excludeAnnotationId ?? null;
  const speciesByAlias = new Map();
  species.forEach(item => taxonomyAliases(item).forEach(alias => {
    if (!speciesByAlias.has(alias)) speciesByAlias.set(alias, item.id);
  }));
  return examples.flatMap(example => {
    if (example.featureVersion !== featureVersion || example.sourceAnnotationId === excludeAnnotationId || !Array.isArray(example.features) || !example.features.length) return [];
    const speciesId = taxonomyAliases(example.species).map(alias => speciesByAlias.get(alias)).find(Boolean);
    return speciesId ? [{ speciesId, features: example.features }] : [];
  });
}

export function predictKnn(samples, features, options = {}) {
  const minSamples = options.minSamples ?? 4, minClasses = options.minClasses ?? 2, k = options.k ?? 7;
  if (!Array.isArray(features) || !features.length || samples.length < minSamples) return null;
  const classes = new Set(samples.map(sample => sample.speciesId));
  if (classes.size < minClasses) return null;
  const compatible = samples.filter(sample => Array.isArray(sample.features) && sample.features.length === features.length);
  if (compatible.length < minSamples) return null;
  const nearest = compatible.map(sample => {
    const distance = Math.sqrt(features.reduce((sum, value, index) => sum + (value - sample.features[index]) ** 2, 0) / features.length);
    return { speciesId: sample.speciesId, distance };
  }).sort((a, b) => a.distance - b.distance).slice(0, Math.min(k, compatible.length));
  const scores = new Map();
  nearest.forEach(item => scores.set(item.speciesId, (scores.get(item.speciesId) || 0) + 1 / (item.distance + 0.025)));
  const ranked = [...scores].sort((a, b) => b[1] - a[1]), total = ranked.reduce((sum, item) => sum + item[1], 0);
  return { speciesId: ranked[0][0], confidence: Number((ranked[0][1] / total).toFixed(4)), neighbours: nearest.length };
}

function dosDateTime(date = new Date()) {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1),
    date: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate()
  };
}

function joinBytes(parts) {
  const length = parts.reduce((sum, part) => sum + part.length, 0);
  const output = new Uint8Array(length);
  let offset = 0;
  parts.forEach(part => { output.set(part, offset); offset += part.length; });
  return output;
}

export class ZipStore {
  constructor() {
    this.files = [];
    this.encoder = new TextEncoder();
  }

  add(name, data) {
    if (!name || name.startsWith('/') || name.includes('..')) throw new Error('ZIP entry names must be relative and may not traverse directories.');
    const bytes = typeof data === 'string' ? this.encoder.encode(data) : data;
    if (Object.prototype.toString.call(bytes) !== '[object Uint8Array]') throw new TypeError('ZIP entries must be strings or Uint8Array instances.');
    this.files.push({ name, data: new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) });
  }

  build() {
    if (this.files.length > 65535) throw new Error('ZIP64 is not supported; reduce the number of exported files.');
    const body = [], central = [];
    let offset = 0;
    const stamp = dosDateTime();

    this.files.forEach(file => {
      const name = this.encoder.encode(file.name), size = file.data.length, crc = crc32(file.data);
      if (size > 0xffffffff || offset > 0xffffffff) throw new Error('ZIP64 is not supported; export a smaller dataset batch.');

      const local = new Uint8Array(30 + name.length), localView = new DataView(local.buffer);
      localView.setUint32(0, 0x04034b50, true); localView.setUint16(4, 20, true); localView.setUint16(6, 0x0800, true);
      localView.setUint16(8, 0, true); localView.setUint16(10, stamp.time, true); localView.setUint16(12, stamp.date, true);
      localView.setUint32(14, crc, true); localView.setUint32(18, size, true); localView.setUint32(22, size, true);
      localView.setUint16(26, name.length, true); localView.setUint16(28, 0, true); local.set(name, 30);

      const directory = new Uint8Array(46 + name.length), directoryView = new DataView(directory.buffer);
      directoryView.setUint32(0, 0x02014b50, true); directoryView.setUint16(4, 20, true); directoryView.setUint16(6, 20, true);
      directoryView.setUint16(8, 0x0800, true); directoryView.setUint16(10, 0, true); directoryView.setUint16(12, stamp.time, true);
      directoryView.setUint16(14, stamp.date, true); directoryView.setUint32(16, crc, true); directoryView.setUint32(20, size, true);
      directoryView.setUint32(24, size, true); directoryView.setUint16(28, name.length, true); directoryView.setUint16(30, 0, true);
      directoryView.setUint16(32, 0, true); directoryView.setUint16(34, 0, true); directoryView.setUint16(36, 0, true);
      directoryView.setUint32(38, 0, true); directoryView.setUint32(42, offset, true); directory.set(name, 46);

      body.push(local, file.data); central.push(directory); offset += local.length + size;
    });

    const centralSize = central.reduce((sum, part) => sum + part.length, 0);
    if (centralSize > 0xffffffff || offset > 0xffffffff) throw new Error('ZIP64 is not supported; export a smaller dataset batch.');
    const end = new Uint8Array(22), endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true); endView.setUint16(4, 0, true); endView.setUint16(6, 0, true);
    endView.setUint16(8, this.files.length, true); endView.setUint16(10, this.files.length, true);
    endView.setUint32(12, centralSize, true); endView.setUint32(16, offset, true); endView.setUint16(20, 0, true);

    return joinBytes([...body, ...central, end]);
  }
}
