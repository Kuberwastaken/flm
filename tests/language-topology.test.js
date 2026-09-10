import test from 'node:test';
import assert from 'node:assert/strict';
import { matchedValidation } from '../web/language-topology.js';

test('topology display compares matching updates and training seeds only', () => {
  const report = { runs: [
    { reference: true, seed: 42, validation: [{ step: 500, bits_per_byte: 2.5 }, { step: 6000, bits_per_byte: 1.9 }] },
    { reference: true, seed: 43, validation: [{ step: 500, bits_per_byte: 2.6 }] },
    { reference: false, seed: 42, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }, { step: 1000, bits_per_byte: 2.2 }] },
    { reference: false, seed: 43, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }] },
    { reference: false, seed: 42, saved_step: 0, validation: [] }
  ] };
  const rows = matchedValidation(report);
  assert.equal(rows[0].point.step, 500);
  assert.equal(rows[0].reference.bits_per_byte, 2.5);
  assert.ok(Math.abs(rows[0].difference - .1) < 1e-12);
  assert.ok(Math.abs(rows[1].difference - .2) < 1e-12);
  assert.equal(rows[2].difference, null);
});

test('topology display leaves unmatched reference measurements empty', () => {
  const rows = matchedValidation({ runs: [
    { reference: true, seed: 42, validation: [{ step: 6000, bits_per_byte: 1.9 }] },
    { reference: false, seed: 42, saved_step: 500, validation: [{ step: 500, bits_per_byte: 2.4 }] }
  ] });
  assert.equal(rows[0].point, undefined);
  assert.equal(rows[0].difference, null);
});
