# NeuralPaper.ai Frontend

This is the frontend for NeuralPaper.ai, a platform that integrates Neural DSL with NeuralDbg to create interactive, annotated neural network models.

## Deployment on Vercel

### Prerequisites

- A [Vercel](https://vercel.com) account
- Git repository with your NeuralPaper.ai code

### Steps to Deploy

1. **Sign up for Vercel**:
   - Go to [vercel.com](https://vercel.com) and sign up for an account
   - Connect your GitHub/GitLab account

2. **Import Your Repository**:
   - Click "Import Project"
   - Select "Import Git Repository"
   - Select your repository with NeuralPaper.ai code

3. **Configure the Project**:
   - Name: `neuralpaper` (or any name you prefer)
   - Framework Preset: `Next.js`
   - Root Directory: `neuralpaper/frontend` (important!)
   - Build Command: `npm run build`
   - Output Directory: `.next`

4. **Environment Variables**:
   - Add the following environment variables:
     - `NEXT_PUBLIC_API_URL`: URL of your backend API (e.g., `https://neuralpaper-api.onrender.com`)

5. **Deploy**:
   - Click "Deploy"
   - Vercel will automatically build and deploy your frontend

### Automatic Deployments

Vercel automatically deploys your frontend when you push changes to your repository. You can configure this behavior in the project settings.

### Accessing Your Frontend

Once deployed, your frontend will be available at:
```
https://neuralpaper.vercel.app
```

### Custom Domain

To use a custom domain:
1. Go to your project settings in Vercel
2. Click "Domains"
3. Add your domain and follow the instructions

## Local Development

To run the frontend locally:

```bash
cd neuralpaper/frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.
