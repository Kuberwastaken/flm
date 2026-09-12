// Targeted browser checks; Playwright is an optional local QA dependency.
// FLM_PREVIEW_URL can point to a local preview or the published site.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const base = process.env.FLM_PREVIEW_URL || 'http://127.0.0.1:5181/';
  const out = path.resolve('work/presentation-review');
  await fs.mkdir(out, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.FLM_BROWSER_CHANNEL || 'chrome' });
  const results = { base, themes: [], videos: [] };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.goto(base + '#research');
    await page.locator('#model option').first().waitFor({ state: 'attached' });
    for (const theme of ['light', 'dark']) {
      if (await page.locator('html').getAttribute('data-theme') !== theme) await page.locator('#theme-toggle').click();
      await page.evaluate(() => { location.hash = 'research'; });
      await page.locator('.architecture-figure').scrollIntoViewIfNeeded();
      const audit = await page.evaluate(() => {
        const rgb = value => value.match(/[\d.]+/g).slice(0, 3).map(Number);
        const luminance = value => rgb(value).map(n => n / 255).map(n => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4).reduce((sum, n, i) => sum + n * [.2126, .7152, .0722][i], 0);
        const contrast = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
        const background = element => {
          for (let node = element; node; node = node.parentElement) {
            const color = getComputedStyle(node).backgroundColor;
            if (!color.endsWith(', 0)') && color !== 'transparent') return color;
          }
          return getComputedStyle(document.body).backgroundColor;
        };
        const diagram = document.querySelector('.architecture-figure text');
        return {
          scheme: getComputedStyle(document.documentElement).colorScheme,
          architectureContrast: contrast(getComputedStyle(diagram).fill, background(diagram)),
          options: [...document.querySelectorAll('select')].map(select => {
            const option = select.querySelector('option:not(:disabled)');
            return { id: select.id, controlContrast: contrast(getComputedStyle(select).color, background(select)),
              optionContrast: option ? contrast(getComputedStyle(option).color, getComputedStyle(option).backgroundColor) : null };
          })
        };
      });
      assert.equal(audit.scheme, theme);
      assert(audit.architectureContrast >= 4.5, `${theme}: architecture text contrast`);
      for (const option of audit.options) {
        assert(option.controlContrast >= 4.5, `${theme}: ${option.id} control contrast`);
        assert(option.optionContrast === null || option.optionContrast >= 4.5, `${theme}: ${option.id} option contrast`);
      }
      await page.locator('.architecture-figure').screenshot({ path: path.join(out, `architecture-${theme}.png`) });
      await page.evaluate(() => { location.hash = 'chat'; });
      await page.locator('#model').click();
      await page.screenshot({ path: path.join(out, `model-picker-${theme}.png`) });
      await page.keyboard.press('Escape');
      results.themes.push({ theme, ...audit });
    }
    await page.goto(base + '#behavior-video');
    await page.waitForFunction(() => document.body.dataset.page === 'research');
    assert(await page.locator('#behavior-video').isVisible(), 'README video link opens research');
    // Exercise both actual codecs, rather than accepting a poster or canPlayType().
    for (const name of ['closed-loop', 'learned-choice', 'physical-calibration']) {
      for (const extension of ['webm', 'mp4']) {
        const result = await page.evaluate(async ({ name, extension }) => {
          const video = document.querySelector('#behavior-video');
          video.pause(); video.src = `/research/${name}.${extension}`; video.muted = true; video.load();
          await video.play();
          await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Playback did not advance')), 10000);
            const check = () => { if (video.currentTime > .15 && video.videoWidth > 0) { clearTimeout(timeout); video.removeEventListener('timeupdate', check); resolve(); } };
            video.addEventListener('timeupdate', check); check();
          });
          const result = { name, extension, currentTime: video.currentTime, duration: video.duration,
            width: video.videoWidth, height: video.videoHeight, decodedFrames: video.getVideoPlaybackQuality().totalVideoFrames };
          video.pause(); return result;
        }, { name, extension });
        assert(result.decodedFrames > 0, 'Video must decode actual frames');
        results.videos.push(result);
      }
    }
    await page.goto(base + '#behavior-video');
    await page.reload();
    await page.locator('#behavior-video').scrollIntoViewIfNeeded();
    await page.locator('#behavior-video').evaluate(video => { video.muted = true; return video.play(); });
    await page.waitForFunction(() => document.querySelector('#behavior-video').currentTime > .2);
    assert((await page.locator('#behavior-video').evaluate(video => video.currentSrc)).endsWith('/closed-loop.webm'));
    await page.locator('#behavior-video').screenshot({ path: path.join(out, 'video-playing.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(base + '#chat');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile chat should not overflow horizontally');
    await page.screenshot({ path: path.join(out, 'mobile-chat.png') });
    await fs.writeFile(path.join(out, 'checks.json'), JSON.stringify(results, null, 2) + '\n');
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
