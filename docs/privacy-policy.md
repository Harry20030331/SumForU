# SumForU Privacy Policy

Last updated: July 1, 2026

SumForU is a Chrome extension that creates short, persona-aware buying summaries for product pages.

## Data We Process

When you click Generate, SumForU may process:

- Text extracted from the active product page, such as product title, price, ratings, reviews, specs, warranty, and availability details.
- The shopper persona you enter in the extension.
- The current page URL and hostname.
- An anonymous install ID used for rate limiting and abuse prevention.

SumForU does not collect your full browsing history, payment information, passwords, account credentials, or personal documents.

## How We Use Data

We use the data above only to:

- Generate the product buying summary you requested.
- Personalize the recommendation to your stated shopper persona.
- Operate the backend service and prevent abuse through basic rate limiting.

## Data Sharing

To generate a summary, SumForU sends the extracted page evidence and shopper persona to the SumForU backend API. The backend calls OpenAI to produce the summary.

We do not sell user data. We do not use user data for advertising. We do not share user data for unrelated purposes.

## Data Storage

The extension stores your shopper persona, backend URL setting, and anonymous install ID locally in Chrome storage.

The backend is designed to process summary requests transiently. It does not intentionally store product page text or shopper personas in an application database.

## User Control

You can edit or clear your shopper persona from the extension settings. You can uninstall the extension at any time to remove locally stored extension data from Chrome.

## Security

The OpenAI API key is kept on the backend and is not included in the browser extension. Requests to the backend use HTTPS.

## Contact

For questions about this privacy policy, contact the project owner through the GitHub repository:

https://github.com/Harry20030331/SumForU
