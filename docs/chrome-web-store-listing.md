# Sum for You Chrome Web Store Listing

## Product Details

Name:
Sum for You

Summary:
Personalized buying summaries for product pages, grounded in reviews, price, and page details.

Category:
Shopping

Language:
English

Detailed description:
Sum for You helps you decide whether a product fits you before you buy.

Open a product page, describe what you care about once, and click Generate. Sum for You reads the current page and creates a short buying fit check with a recommendation, score, key strengths, and key concerns.

It is designed for review-heavy product pages where the important tradeoffs are scattered across ratings, customer comments, price, specs, warranty, and availability details.

What it does:
- Summarizes the current product page into a compact recommendation
- Uses your saved shopper persona to personalize the tradeoffs
- Highlights the most important strengths and concerns
- Keeps the OpenAI API key on the backend, not in the browser extension

What it does not do:
- It does not make purchases
- It does not change website content
- It does not collect browsing history
- It does not ask users to provide an OpenAI API key

## Graphic Assets

Store icon:
extension/assets/icons/icon-128.png

Screenshot:
store-assets/screenshot-1280x800.png

Small promo tile:
store-assets/small-promo-440x280.png

Marquee promo tile:
store-assets/marquee-promo-1400x560.png

## Privacy Practices

Single purpose:
Sum for You creates a short, persona-aware buying summary for the product page the user is currently viewing.

Permission justification:

activeTab:
Used only after the user opens the extension and clicks Generate, so Sum for You can read the current product page.

scripting:
Used to inject the bundled page-reading script into the active tab after the user clicks Generate.

storage:
Used to save the user's shopper persona, backend URL setting, and an anonymous install ID for rate limiting.

Remote code:
No remote code is executed. The extension only runs bundled JavaScript. It sends extracted page text to the Sum for You backend API to request a generated summary.

Data collection disclosure:
Sum for You processes website content from the active product page, the user's shopper persona, and an anonymous install ID. This data is used only to generate the requested product summary and enforce basic abuse prevention/rate limiting.

Data use certification:
Data is not sold, used for unrelated advertising, or transferred for purposes unrelated to the extension's single purpose.

Privacy policy URL:
Use the GitHub-rendered privacy policy after pushing:
https://github.com/Harry20030331/SumForU/blob/main/docs/privacy-policy.md

## Distribution

Visibility:
Public

Regions:
All regions

Pricing:
Free

Mature content:
No

## Review Notes

Test instructions:
1. Install the extension.
2. Open a product page with reviews, for example a product page on Best Buy.
3. Click the Sum for You extension icon.
4. Enter a shopper persona, or use the default persona.
5. Click Generate.
6. Confirm the popup displays a recommendation, score, strengths, and concerns.

Backend:
https://sumforu-api.onrender.com
