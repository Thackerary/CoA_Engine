# CoA_Engine Architecture & Coding Conventions

1. **Language & Build**: C++20 (`std::span`, `std::optional`, `[[nodiscard]]`). Compiler is MSVC (Visual Studio 2022) with `/utf-8`.
2. **OpenGL 4.6 Core DSA Only**:
   - STRICTLY use Direct State Access (DSA): `glCreateBuffers`, `glNamedBufferStorage`, `glNamedBufferSubData`, `glCreateTextures`, `glTextureStorage2D`, `glCompressedTextureSubImage2D`, `glBindTextureUnit`, `glVertexArrayVertexBuffer`.
   - NEVER use legacy state-machine bindings (`glGenBuffers`, `glBindBuffer(GL_ARRAY_BUFFER)`, `glBindTexture`, `glActiveTexture`).
3. **Coordinate System & Depth**:
   - World space is WoW native **Right-Handed Z-Up** (`+X` North, `+Y` West, `+Z` Up).
   - Depth buffer uses **Reverse-Z**: `glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE)`, `glDepthFunc(GL_GREATER)`, clear depth `0.0`.
4. **Binary Parsing Rules**:
   - All WoW 3.3.5a (Build 12340) file headers (`BLP2`, `MD20`, `SKIN`, `MCNK`, `MOHD`, `MOGP`) MUST use `#pragma pack(push, 1)` and `static_assert(sizeof(...))`[cite: 5, 7, 12].
   - File I/O MUST go through `coa::vfs::VirtualFileSystem::ReadFile` and `NormalizeVirtualPath`[cite: 4, 7].