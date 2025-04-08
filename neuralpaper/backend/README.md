# NeuralPaper.ai Backend

This is the backend API for NeuralPaper.ai, a platform that integrates Neural DSL with NeuralDbg to create interactive, annotated neural network models.

## Deployment on Render

### Prerequisites

- A [Render](https://render.com) account
- Git repository with your NeuralPaper.ai code

### Steps to Deploy

1. **Sign up for Render**:
   - Go to [render.com](https://render.com) and sign up for an account
   - Connect your GitHub/GitLab account

2. **Create a New Web Service**:
   - Click "New" and select "Web Service"
   - Connect your repository
   - Select the repository with your NeuralPaper.ai code

3. **Configure the Web Service**:
   - Name: `neuralpaper-api` (or any name you prefer)
   - Environment: `Python 3`
   - Region: Choose the region closest to your users
   - Branch: `main` (or your default branch)
   - Root Directory: `neuralpaper/backend` (important!)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT`

4. **Environment Variables**:
   - Click "Advanced" and add the following environment variables:
     - `NEURAL_ENV`: `production`
     - `PYTHON_VERSION`: `3.9.0` (or your preferred version)

5. **Create Web Service**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your API

### Automatic Deployments

Render automatically deploys your API when you push changes to your repository. You can disable this in the settings if you prefer manual deployments.

### Accessing Your API

Once deployed, your API will be available at:
```
https://neuralpaper-api.onrender.com
```

You can test it by visiting:
```
https://neuralpaper-api.onrender.com/health
```

### Connecting Frontend to Backend

Update your frontend configuration to point to your Render API:

```javascript
// In neuralpaper/frontend/next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://neuralpaper-api.onrender.com/:path*',
      },
    ]
  },
}
```

## Local Development

To run the backend locally:

```bash
cd neuralpaper/backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.
