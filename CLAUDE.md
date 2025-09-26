# CLAUDE.md

This is the Claude Code project template. Copy this file to any new project folder and run `/init` to create a project-specific setup.

## Quick Start

1. Copy this `CLAUDE.md` file to your new project folder
2. Enable essential Claude Code settings for seamless development:
   - `enableAllProjectMcpServers`: Automatically connects all configured MCP servers
   - `dangerouslySkipPermissions`: Bypasses permission checks for smoother development (no prompts for file changes, code edits, or running commands)
3. Run `/init` command in Claude Code to initialize the project
4. Follow the generated project-specific CLAUDE.md instructions

## New Chat Session Setup

**Important**: When starting a new chat session, Claude doesn't automatically read project files or load MCP servers. To ensure proper initialization:

1. **First message in new chat**: Include a brief project context or ask Claude to read the `CLAUDE.md` file
2. **Alternative**: Add this instruction to your first prompt: "Please read the CLAUDE.md file and initialize any configured MCP servers"
3. **Best practice**: Start each session with: "Initialize project based on CLAUDE.md configuration"

---

## Template Overview

This template guides you through setting up a new project with Claude Code in four progressive phases, from basic setup to advanced workflow optimization.

---

## Phase 1: Project Foundation

### 1.1 Repository Initialization

**The `/init` command will create:**

- Project-specific `CLAUDE.md` with customized instructions
- Essential configuration files (`.mcp.json`, `settings.local.json`)
- Git repository initialization
- Technology stack-specific `.gitignore`
- Package management files (`package.json`, `requirements.txt`, etc.)

**Manual Setup (if not using `/init`):**

```bash
git init
git worktree add -b main .
```

**Basic Structure:**

- Create project-specific `CLAUDE.md` based on this template
- Create `.gitignore` for your technology stack
- Initialize package management (`package.json`, `requirements.txt`, etc.)

### 1.2 Claude Code Configuration

**Essential Settings (`settings.local.json`):**

```json
{
  "enableAllProjectMcpServers": true,
  "dangerouslySkipPermissions": true
}
```

**Benefits:**

- `enableAllProjectMcpServers`: Automatically connects all configured MCP servers
- `dangerouslySkipPermissions`: Bypasses permission checks for smoother development

### 1.3 Core MCP Servers Setup

**Memory Server Configuration (`.mcp.json`):**

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "time": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-time"]
    }
  }
}
```

**Memory Server Benefits:**

- Persistent knowledge graph across sessions
- Entity and relationship tracking
- Project context preservation

**Time Server Benefits:**

- Accurate timezone conversions
- Real-time clock functionality
- IANA timezone database support

---

## Phase 2: Development Environment

### 2.1 Development Approach Selection

**Single Instance Development:**

- Use for simple projects or solo development
- All work in main directory
- Linear development workflow

**Parallel Worktrees Development:**

```bash
git worktree add -b feature-x ../feature-x
git worktree add -b api-development ../api-dev
git worktree add -b frontend-work ../frontend
```

**Parallel Benefits:**

- Multiple Claude Code instances on different components
- True parallelization of development work
- Isolated feature development
- Clean integration via pull requests

### 2.2 Model Strategy Selection

**Git Worktree Model Strategy:**

```bash
# Design and architecture phase
git worktree add -b design-phase ../design-opus

# Implementation phase
git worktree add -b implementation ../impl-sonnet
```

**Recommended Workflow:**

- **Opus Phase**: Initial design, architecture planning, complex problem-solving
- **Sonnet Phase**: Implementation, coding, testing, optimization

### 2.3 Specialized MCP Servers (Optional)

**Context7 Server (for library work):**

```json
"context7": {
  "command": "npx",
  "args": ["-y", "context7"]
}
```

- Add "use context7" to prompts when working with libraries
- Provides current, version-specific documentation
- Eliminates outdated API references

**Playwright Server (for browser automation):**

```json
"playwright": {
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest"]
}
```

- Browser automation without screenshots
- Structured accessibility snapshots
- Requires Node.js 18 or newer

---

## Phase 3: Integration Planning

### 3.1 Testing and Process Setup

**Testing Requirements:**

- Write tests for each feature implementation
- Run tests after each feature implementation
- Only write necessary code for the feature

**Process Requirements:**

- Ask clarifying questions (one per prompt) until 95% sure of requirements
- Write detailed design documents
- Separate implementation into stages
- Use detailed todo lists for implementation

### 3.2 Version Control Strategy

**Basic Git Workflow:**

- Use git to capture progress after each major change
- Use branches for features
- Use commits for major updates

**Advanced Git Strategy:**

- Feature branches for isolated development
- Pull request workflow for code review
- Integration branches for complex features

### 3.3 Claude Code Features Integration

**Plan Mode:**

- Use `/plan` command for complex multi-step implementations
- Allows review and approval before execution
- Useful for tasks affecting multiple files

**Slash Commands:**

- `/init`: Reads this template and creates project-specific CLAUDE.md with all necessary configuration files
- Custom commands based on project needs

**Using `/init` Command:**

When you run `/init` in a new project folder containing this template, Claude Code will:

1. Analyze the current project structure and detect technology stack
2. Generate a project-specific `CLAUDE.md` with customized instructions
3. Create `.mcp.json` with appropriate MCP server configurations
4. Create `settings.local.json` with optimal Claude Code settings
5. Initialize git repository if not already present
6. Create technology-appropriate `.gitignore` file
7. Set up package management files based on detected stack

### 3.4 GitHub App Integration (Optional)

**Installation Endpoints:**

- **Installation URL**: `https://github.com/apps/{app_name}/installations/new`
- **Installation Token**: `POST /app/installations/{installation_id}/access_tokens`
- **List Installations**: `GET /app/installations`

**Authentication Flow:**

1. Generate JWT for your app
2. Get installation token using JWT
3. Use installation token for repository operations
4. Refresh tokens (expire after 1 hour)

**Benefits Over OAuth Apps:**

- Fine-grained permissions
- User control over repository access
- Short-lived tokens
- Enterprise-level installation support

---

## Phase 4: Workflow Optimization

### 4.1 SuperClaude Framework Integration (Advanced)

**Key Features:**

- **Specialized Agents**: 15 domain experts (security, frontend, research, etc.)
- **Deep Research Mode**: Autonomous multi-hop web research with quality scoring
- **MCP Server Integration**: 7 specialized servers for web search, content extraction
- **Structured Command System**: 25 slash commands with "/sc:" prefix

**Applications:**

- Code Reviews: Security engineer agent for vulnerability analysis
- Architecture Planning: Frontend/backend architect agents for design decisions
- Research Tasks: Deep research mode for technology evaluation
- Documentation: Technical documentation lookup and content extraction

**Repository**: [SuperClaude Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework)

### 4.2 Enterprise Features (2025)

**GitHub Apps Enterprise Access:**

- Install apps on enterprise accounts
- Access enterprise management APIs (public preview)
- Organization Installations REST API for enterprise control
- Install apps onto organizations within an enterprise

### 4.3 Memory System Advanced Usage

**Entity Types:**

- **user**: Primary users of the system
- **person**: Individuals (colleagues, contacts)
- **organization**: Companies, teams, groups
- **project**: Development projects, features
- **concept**: Ideas, technologies, methodologies

**Usage Patterns:**

- "Create an entity for [name]"
- "Add information about [entity]"
- "Create a relationship between [entity1] and [entity2]"
- "Search for entities related to [topic]"

---

## Implementation Checklist

### Phase 1 Checklist

- [ ] Initialize git repository
- [ ] Create `settings.local.json` with core settings
- [ ] Set up Memory and Time MCP servers
- [ ] Test basic Claude Code functionality

### Phase 2 Checklist

- [ ] Choose development approach (single/parallel)
- [ ] Set up worktrees if using parallel development
- [ ] Configure additional MCP servers based on project needs
- [ ] Establish model switching strategy

### Phase 3 Checklist

- [ ] Set up testing framework
- [ ] Establish version control workflow
- [ ] Configure GitHub App if needed
- [ ] Implement process requirements

### Phase 4 Checklist

- [ ] Evaluate SuperClaude Framework integration
- [ ] Set up enterprise features if applicable
- [ ] Optimize memory system usage
- [ ] Document final workflow patterns

---

## Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary for achieving your goal
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files unless explicitly requested
