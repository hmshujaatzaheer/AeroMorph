# ============================================================================
#                    AEROMORPH GITHUB REPOSITORY SETUP GUIDE
#                    Step-by-Step PowerShell Commands
# ============================================================================
#
# Author: H M Shujaat Zaheer
# Email: shujabis@gmail.com
# GitHub: https://github.com/hmshujaatzaheer
#
# This guide teaches you EVERY SINGLE STEP to upload the AeroMorph repository
# to GitHub, explained like teaching a kindergarten student.
#
# ============================================================================

# ============================================================================
#                           PART 0: PREREQUISITES
# ============================================================================
# Before starting, you need:
#   1. Git installed on your computer
#   2. A GitHub account (https://github.com)
#   3. The AeroMorph.zip file downloaded and extracted
#
# To check if Git is installed, open PowerShell and type:
# ----------------------------------------------------------------------------

git --version

# If you see a version number (like "git version 2.40.0"), Git is installed!
# If you get an error, download Git from: https://git-scm.com/download/win


# ============================================================================
#                    PART 1: CONFIGURE GIT (ONE-TIME SETUP)
# ============================================================================
# These commands tell Git who you are. Do this ONLY ONCE on your computer.
# ----------------------------------------------------------------------------

# Step 1.1: Set your name (replace with your actual name)
# Think of this like writing your name on your homework
git config --global user.name "H M Shujaat Zaheer"

# Step 1.2: Set your email (MUST match your GitHub email!)
# This is like putting your email address on a letter
git config --global user.email "shujabis@gmail.com"

# Step 1.3: Verify your settings (optional - just to check)
# This shows you what you just set
git config --global --list


# ============================================================================
#                    PART 2: EXTRACT AND NAVIGATE TO FOLDER
# ============================================================================
# First, download and extract the AeroMorph.zip file
# ----------------------------------------------------------------------------

# Step 2.1: Open File Explorer and extract AeroMorph.zip
# Right-click AeroMorph.zip → "Extract All" → Choose a location (e.g., Desktop)

# Step 2.2: Open PowerShell and navigate to the AeroMorph folder
# (Change the path below to match where you extracted it)

# If you extracted to Desktop:
cd "$HOME\Desktop\AeroMorph"

# OR if you extracted to Downloads:
# cd "$HOME\Downloads\AeroMorph"

# Step 2.3: Verify you're in the right folder
# You should see files like README.md, setup.py, etc.
dir

# Expected output should show:
#   aeromorph/
#   docs/
#   examples/
#   tests/
#   README.md
#   setup.py
#   ... etc.


# ============================================================================
#                    PART 3: CREATE GITHUB REPOSITORY (ONLINE)
# ============================================================================
# You need to create an empty repository on GitHub first
# ----------------------------------------------------------------------------

# Step 3.1: Open your web browser and go to: https://github.com/new
#
# Step 3.2: Fill in the form:
#   - Repository name: AeroMorph
#   - Description: Unified Perception-Driven Morphological Adaptation Framework for Multi-Modal Aerial Robots
#   - Select: Public (so everyone can see your research)
#   - DO NOT check "Add a README file" (we already have one)
#   - DO NOT check "Add .gitignore" (we already have one)
#   - DO NOT check "Choose a license" (we already have one)
#
# Step 3.3: Click the green "Create repository" button
#
# Step 3.4: You'll see a page with instructions - we'll use these next!


# ============================================================================
#                    PART 4: INITIALIZE LOCAL GIT REPOSITORY
# ============================================================================
# Now we set up Git in your AeroMorph folder
# ----------------------------------------------------------------------------

# Step 4.1: Initialize Git in this folder
# This is like creating a magic box that tracks all changes
git init

# You should see: "Initialized empty Git repository in .../AeroMorph/.git/"

# Step 4.2: Check the status (optional - just to see what's happening)
git status

# You should see a list of "Untracked files" in red
# This means Git sees the files but isn't tracking them yet


# ============================================================================
#                    PART 5: ADD FILES TO GIT
# ============================================================================
# Now we tell Git to track all our files
# ----------------------------------------------------------------------------

# Step 5.1: Add ALL files to Git's staging area
# The "." means "everything in this folder"
# Think of this like putting all your papers into an envelope
git add .

# Step 5.2: Check the status again (optional)
git status

# Now you should see files in green under "Changes to be committed"
# This means Git is ready to save these files


# ============================================================================
#                    PART 6: CREATE YOUR FIRST COMMIT
# ============================================================================
# A commit is like taking a snapshot of all your files at this moment
# ----------------------------------------------------------------------------

# Step 6.1: Create your first commit with a message
# The message describes what this snapshot contains
git commit -m "Initial commit: AeroMorph framework with P2MA algorithm, spatial feasibility, and swarm coordination"

# You should see output showing how many files were committed
# Example: "35 files changed, 4500 insertions(+)"


# ============================================================================
#                    PART 7: CONNECT TO GITHUB
# ============================================================================
# Now we connect your local folder to your GitHub repository
# ----------------------------------------------------------------------------

# Step 7.1: Rename the default branch to "main"
# GitHub now uses "main" as the default branch name
git branch -M main

# Step 7.2: Add the GitHub repository as a "remote"
# This tells Git where to upload your files
# REPLACE "hmshujaatzaheer" with your actual GitHub username if different
git remote add origin https://github.com/hmshujaatzaheer/AeroMorph.git

# Step 7.3: Verify the remote was added (optional)
git remote -v

# You should see:
#   origin  https://github.com/hmshujaatzaheer/AeroMorph.git (fetch)
#   origin  https://github.com/hmshujaatzaheer/AeroMorph.git (push)


# ============================================================================
#                    PART 8: PUSH TO GITHUB (UPLOAD!)
# ============================================================================
# This is the moment where your files go to GitHub!
# ----------------------------------------------------------------------------

# Step 8.1: Push (upload) your files to GitHub
# "-u" remembers the settings so next time you just type "git push"
git push -u origin main

# IMPORTANT: A login window may appear!
# - If a browser opens: Log in to GitHub and authorize
# - If a terminal prompt asks: Enter your GitHub username and Personal Access Token
#   (NOT your password - GitHub requires tokens now)

# If successful, you'll see output like:
#   Enumerating objects: 75, done.
#   Counting objects: 100% (75/75), done.
#   ...
#   Branch 'main' set up to track remote branch 'main' from 'origin'.


# ============================================================================
#                    PART 9: VERIFY ON GITHUB
# ============================================================================
# Let's make sure everything uploaded correctly!
# ----------------------------------------------------------------------------

# Step 9.1: Open your browser and go to:
#   https://github.com/hmshujaatzaheer/AeroMorph
#
# Step 9.2: You should see all your files!
#   - README.md displayed nicely
#   - All folders (aeromorph/, docs/, examples/, tests/)
#   - All configuration files


# ============================================================================
#                    PART 10: CREATING A PERSONAL ACCESS TOKEN
# ============================================================================
# If GitHub asks for authentication, you need a Personal Access Token
# (GitHub no longer accepts passwords for command line)
# ----------------------------------------------------------------------------

# Step 10.1: Go to: https://github.com/settings/tokens
#
# Step 10.2: Click "Generate new token" → "Generate new token (classic)"
#
# Step 10.3: Fill in:
#   - Note: "AeroMorph access" (or any name you want)
#   - Expiration: "90 days" or "No expiration"
#   - Check these scopes:
#     ✅ repo (all checkboxes under it)
#     ✅ workflow
#
# Step 10.4: Click "Generate token" at the bottom
#
# Step 10.5: COPY THE TOKEN IMMEDIATELY!
#   - It looks like: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   - You can only see it once!
#   - Save it somewhere safe
#
# Step 10.6: When Git asks for password, paste this token instead


# ============================================================================
#                    PART 11: MAKING FUTURE CHANGES
# ============================================================================
# After your initial upload, here's how to add more changes:
# ----------------------------------------------------------------------------

# Step 11.1: Make your changes to files

# Step 11.2: See what files changed
git status

# Step 11.3: Add the changed files
git add .

# Step 11.4: Create a new commit with a descriptive message
git commit -m "Add new feature: predictive morphing algorithm"

# Step 11.5: Push to GitHub
git push

# That's it! Your changes are now on GitHub!


# ============================================================================
#                    TROUBLESHOOTING COMMON ISSUES
# ============================================================================

# ISSUE 1: "git is not recognized as a command"
# SOLUTION: Install Git from https://git-scm.com/download/win
#           Then restart PowerShell

# ISSUE 2: "Permission denied" or "Authentication failed"
# SOLUTION: Create a Personal Access Token (see Part 10)
#           Use the token as your password

# ISSUE 3: "remote origin already exists"
# SOLUTION: Remove the old remote and add new one:
#   git remote remove origin
#   git remote add origin https://github.com/hmshujaatzaheer/AeroMorph.git

# ISSUE 4: "failed to push some refs"
# SOLUTION: Pull first, then push:
#   git pull origin main --rebase
#   git push

# ISSUE 5: "fatal: not a git repository"
# SOLUTION: Make sure you're in the AeroMorph folder:
#   cd path\to\AeroMorph
#   git init


# ============================================================================
#                    QUICK REFERENCE CARD
# ============================================================================
# Most common commands you'll use:
#
#   git status          → See what changed
#   git add .           → Stage all changes
#   git commit -m "msg" → Save changes with message
#   git push            → Upload to GitHub
#   git pull            → Download from GitHub
#   git log --oneline   → See commit history
#
# ============================================================================

# ============================================================================
#                         ALL COMMANDS IN ORDER
# ============================================================================
# Copy and paste these commands one by one:
# ----------------------------------------------------------------------------

# 1. Configure Git (one-time):
git config --global user.name "H M Shujaat Zaheer"
git config --global user.email "shujabis@gmail.com"

# 2. Navigate to folder:
cd "$HOME\Desktop\AeroMorph"

# 3. Initialize repository:
git init

# 4. Add all files:
git add .

# 5. Create first commit:
git commit -m "Initial commit: AeroMorph framework with P2MA algorithm, spatial feasibility, and swarm coordination"

# 6. Rename branch:
git branch -M main

# 7. Add remote:
git remote add origin https://github.com/hmshujaatzaheer/AeroMorph.git

# 8. Push to GitHub:
git push -u origin main

# DONE! 🎉

# ============================================================================
#                              END OF GUIDE
# ============================================================================
