// Targeted preview-release QA; Playwright is an optional local dependency.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const crypto = require('node:crypto');

(async () => {
  const base = process.env.FLM_PREVIEW_URL || 'http://127.0.0.1:5181/';
  const browser = await chromium.launch({headless:true, channel:'chrome'});
  const results = {format:'flm-selection-preview-browser-review-v1', base, browser:browser.version(), models:[]};
  await fs.mkdir('work/selection-preview-browser', {recursive:true});
  try {
    for (const [id,neurons,theme] of [['flm-kc-l-s42',487,'light'],['flm-kc-l-s43',487,'dark'],['flm-kc-r-s42',540,'light'],['flm-kc-r-s43',540,'dark']]) {
      const page = await browser.newPage({viewport:{width:1440,height:1000},colorScheme:theme});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}?model=${id}#chat`);
      await page.waitForFunction(n => document.querySelector('#neuron').options.length===n, neurons);
      await page.waitForFunction(() => !document.querySelector('#generate').disabled);
      assert.match(await page.locator('#capability').textContent(),/held-out evaluation pending/);
      assert.match(await page.locator('#model-status').textContent(),/preview/);
      assert.equal(await page.locator('#model option').count(),23);
      assert.equal(await page.locator('#typing-fly canvas').count(),1);
      const before = await page.locator('#brain-view').screenshot();
      await page.locator('#prompt').fill('The little bird returned to the garden.');
      await page.evaluate(() => {document.querySelector('#limit').value='32'; document.querySelector('#observe').checked=false;});
      await page.locator('#generate').click();
      await page.waitForFunction(() => !document.querySelector('#generate').disabled && document.querySelectorAll('.message.model').length>0);
      const answer = await page.locator('.message.model .message-text').last().textContent();
      assert(answer.length>0);
      assert.match(await page.locator('#activity-value').textContent(),/observed updates/);
      assert.match(await page.locator('#anatomy-note').textContent(),new RegExp(String(neurons)));
      const after = await page.locator('#brain-view').screenshot();
      assert(!before.equals(after),'Actual generated state changes the brain rendering');
      await page.screenshot({path:`work/selection-preview-browser/${id}-${theme}.png`});
      await page.locator('#mode').selectOption('completion');
      assert.match(await page.locator('#capability').textContent(),/BabyLM 10M/);
      assert(!/WikiText/.test(await page.locator('#capability').textContent()));
      await page.setViewportSize({width:390,height:844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors,[]);
      results.models.push({id,neurons,theme,answer,brain_render_changed:true,typing_fly_canvas:true,mobile_overflow:false,
        before_sha256:crypto.createHash('sha256').update(before).digest('hex'),after_sha256:crypto.createHash('sha256').update(after).digest('hex'),page_errors:errors});
      await page.close();
    }
    const page = await browser.newPage();
    await page.goto(base);
    await page.waitForFunction(() => !document.querySelector('#generate').disabled);
    results.default_model = await page.locator('#model').inputValue();
    assert.equal(results.default_model,'babylm-100m-flm-s43');
    results.checked_at_utc = new Date().toISOString();
    results.scope = 'Browser inference, anatomy, preview disclosure, themes and responsive layout. Generated diagnostic outputs are unedited; not a language-quality benchmark.';
    await fs.writeFile('reports/selection-language/mac-v1/browser-preview-review-v1.json',JSON.stringify(results,null,2)+'\n');
    console.log(JSON.stringify(results,null,2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
