# 🗣️ German Pronunciation Practice – Next.js Frontend

This is a modern Next.js 14 app for practicing and improving your German pronunciation. It connects to a backend AI API for automatic speech recognition, pronunciation scoring, and personalized feedback.

---

## Features
- 🎤 Record your voice reading a German sentence
- 📝 See instant transcript, pronunciation score, and mistake highlights
- 👩‍🏫 Get AI-powered teacher feedback and improvement tips
- 🔄 Request new sentences to practice
- 🌙 Responsive, dark-mode ready UI (Chakra UI)

---

## How It Works
1. The app displays a German sentence for you to read aloud.
2. You record your voice in-browser (Web Audio API).
3. The audio and sentence are sent to the backend API (`/api/evaluate`).
4. The backend transcribes your speech, scores your pronunciation, and returns feedback.
5. The frontend displays your transcript, score, mistakes, and teacher feedback.

---

## Getting Started

### 1. Install dependencies
```bash
npm install
# or
yarn install
```

### 2. Set up environment variables
Create a `.env.local` file in the `frontend/` directory with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
AZURE_OPENAI_API_VERSION= your preview
```

### 3. Start the development server
```bash
npm run dev
# or
yarn dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

> **Note:** The backend API (Flask, port 8000) must be running for evaluation to work. See the main project README for backend setup.

---

## Project Structure
- `src/app/page.tsx` – Main page logic (recording, API calls, state)
- `src/components/RecordingCard.tsx` – UI for recording and sentence display
- `src/components/ScoreFeedbackCard.tsx` – UI for score, transcript, mistakes, and feedback
- `src/app/api/evaluate/route.ts` – Next.js API route that proxies requests to the backend and Azure OpenAI

---

## Customization
- Sentences are fetched from `/api/sentence` (see backend for details)
- UI built with Chakra UI, easily themeable
- Teacher feedback powered by Azure OpenAI (customizable prompt in `route.ts`)

---

## Deployment
This app can be deployed to Vercel or any Node.js hosting. Make sure to set the required environment variables and ensure the backend API is accessible.

---

## License
MIT License © 2025
