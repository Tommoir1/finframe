import test from 'node:test';
import assert from 'node:assert/strict';
import { ZipStore, cocoBoxFromNormalized, crc32, csvText, yoloLineFromNormalized } from '../dataset-utils.js';

test('COCO geometry converts normalised top-left boxes to pixels', () => {
  assert.deepEqual(cocoBoxFromNormalized({ x: 0.125, y: 0.25, w: 0.5, h: 0.25 }, 1920, 1080), [240, 270, 960, 270]);
});

test('YOLO geometry uses normalised box centre coordinates', () => {
  assert.equal(yoloLineFromNormalized(3, { x: 0.1, y: 0.2, w: 0.4, h: 0.2 }), '3 0.300000 0.300000 0.400000 0.200000');
});

test('CSV output quotes commas and embedded quotes', () => {
  assert.equal(csvText([['species', 'note'], ['Snapper', 'two, "large" fish']]), '"species","note"\r\n"Snapper","two, ""large"" fish"');
});

test('CRC32 matches the standard check value', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});

test('ZIP store writes local, central-directory and end signatures', async () => {
  const zip = new ZipStore();
  zip.add('labels/frame.txt', '1 0.5 0.5 0.2 0.2');
  const bytes = zip.build();
  const view = new DataView(bytes.buffer);
  assert.equal(view.getUint32(0, true), 0x04034b50);
  assert.equal(view.getUint32(bytes.length - 22, true), 0x06054b50);
  assert.match(new TextDecoder().decode(bytes), /labels\/frame\.txt/);
});

test('ZIP store rejects path traversal', () => {
  const zip = new ZipStore();
  assert.throws(() => zip.add('../private.txt', 'no'), /relative/);
});
