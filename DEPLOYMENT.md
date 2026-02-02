# 🚀 Vercel Deployment Guide for SignBridge AI

## 📋 Table of Contents
- [Deployment Steps](#deployment-steps)
- [Image Loading Issues & Solutions](#image-loading-issues--solutions)
- [Environment Variables](#environment-variables)
- [Post-Deployment Troubleshooting](#post-deployment-troubleshooting)

---

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository
```bash
# Make sure all changes are committed
git add .
git commit -m "Prepare for Vercel deployment"
git push -u origin main
```

### Step 2: Install Vercel CLI (Optional)
```bash
npm install -g vercel
```

### Step 3: Deploy to Vercel

#### Option A: Using Vercel Dashboard (Easiest)
1. Go to https://vercel.com
2. Sign in with GitHub account
3. Click "Add New..." → "Project"
4. Select your repository: `bidirectional-sign-translator`
5. Configure project settings:
   - **Framework Preset**: Vite
   - **Root Directory**: ./
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
6. Add Environment Variables:
   - **VITE_GEMINI_API_KEY**: Your Gemini API key
7. Click "Deploy"

#### Option B: Using Vercel CLI
```bash
# Login to Vercel
vercel login

# Deploy project
vercel

# For production deployment
vercel --prod
```

---

## 🖼️ Image Loading Issues & Solutions

### ❌ **Problem: Images Not Displaying After Deployment**

**Root Cause:**
The reference images are stored in `src/reference/` but the code references them as `/reference/`. In production, Vite doesn't serve files from `src/` directly—they need to be in the `public/` directory.

### ✅ **Solution: Move Images to Public Directory**

#### **Step 1: Create public directory structure**
```bash
# Create public/reference folder with all subfolders
mkdir -p public/reference/{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}
```

#### **Step 2: Copy all sign language images**
```bash
# Copy all images from src/reference to public/reference
xcopy "src\reference\*" "public\reference\" /S /Y
```

Or manually:
1. Create `public/reference/` folder in your project root
2. Copy all A-Z folders from `reference/` to `public/reference/`

#### **Step 3: Update constants.ts**
The paths should already work since they reference `/reference/` which will serve from `public/reference/` in production.

**Verify constants.ts uses:**
```typescript
export const ASL_ALPHABET: Record<string, string> = {
    a: '/reference/A/250px-Sign_language_A.svg.png',
    b: '/reference/B/250px-Sign_language_B.svg.png',
    // ... etc
};
```

#### **Step 4: Commit and push changes**
```bash
git add public/
git commit -m "Add sign language reference images to public directory"
git push origin main
```

#### **Step 5: Deploy again**
```bash
vercel --prod
```

---

## 🔐 Environment Variables

### For Local Development:
Create `.env.local`:
```
VITE_GEMINI_API_KEY=your_gemini_api_key_here
```

### For Vercel Production:
1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add:
   - **Name**: `VITE_GEMINI_API_KEY`
   - **Value**: Your Gemini API key
   - **Environment**: Production, Preview, Development (select all)
4. Click "Save"

**Do NOT commit `.env.local` to git!** (Already in `.gitignore`)

---

## ✅ Complete Deployment Checklist

- [ ] All files committed and pushed to GitHub
- [ ] `.env.local` is in `.gitignore` (not committed)
- [ ] `public/reference/` folder exists with all images
- [ ] `vite.config.ts` correctly configured
- [ ] `package.json` build scripts defined
- [ ] Environment variable `VITE_GEMINI_API_KEY` set in Vercel
- [ ] Domain configured (optional)
- [ ] HTTPS enabled (automatic on Vercel)

---

## 🐛 Post-Deployment Troubleshooting

### Issue 1: Images Still Not Displaying
**Solution:**
```bash
# Verify images are in public directory
# Go to https://your-vercel-domain/_next/public/reference/A/...

# If not working, try:
1. Clear Vercel cache: Dashboard → Settings → Git → Clear Cache
2. Redeploy: Dashboard → Deployments → Redeploy
```

### Issue 2: API Key Not Working
**Solution:**
```bash
# Verify environment variable is set:
1. Go to Vercel Dashboard → Settings → Environment Variables
2. Check VITE_GEMINI_API_KEY is set
3. Redeploy for changes to take effect
```

### Issue 3: Build Fails
**Common causes:**
```bash
# Missing dependencies
npm install

# TypeScript errors
npm run build

# Check build command works locally first
npm run build
npm run preview
```

### Issue 4: Camera Permission Issues on Mobile
**Solution:**
- Ensure HTTPS is enabled (automatic on Vercel)
- Camera access requires secure context (HTTPS)
- Test on mobile: https://your-domain.vercel.app

---

## 📊 Production Build Optimization

### Before Deploying:
```bash
# Test production build locally
npm run build
npm run preview

# Check bundle size
npm run build -- --stats
```

### Vite Configuration for Production:
Your `vite.config.ts` is already optimized. For additional optimization:
```typescript
build: {
  minify: 'terser',
  sourcemap: false,
  rollupOptions: {
    output: {
      manualChunks: {
        'react': ['react', 'react-dom'],
        'mediapipe': ['@mediapipe/tasks-vision'],
        'gemini': ['@google/genai']
      }
    }
  }
}
```

---

## 🔗 Useful Links

- **Vercel Dashboard**: https://vercel.com/dashboard
- **Gemini API Console**: https://makersuite.google.com/app/apikey
- **Vite Documentation**: https://vitejs.dev/
- **Vercel Docs**: https://vercel.com/docs
- **GitHub Deployment Guide**: https://vercel.com/docs/git

---

## 📞 Quick Support

**If deployment fails:**
1. Check Vercel build logs (Dashboard → Deployments → Failed Build → Logs)
2. Verify environment variables are set
3. Run `npm run build` locally to test
4. Check `.gitignore` doesn't exclude important files
5. Ensure `public/reference/` folder is committed

---

**Your SignBridge AI is ready for production! 🚀**
