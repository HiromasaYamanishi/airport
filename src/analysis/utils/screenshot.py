import os
import asyncio
from pyppeteer import launch

browser = await launch(headless=True)
page = await browser.newPage()
await page.goto(f"/home/yamanishi/project/airport/src/analysis/kagawa1.html")
await page.setViewport({'width': 1080, 'height': 1080})
await page.waitFor(5000)
await page.screenshot({'path': '../data/screenshot/map1.png'})