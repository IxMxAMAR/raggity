# raggity demo tape
# Regenerate the GIF with: vhs demo.tape   (requires VHS: https://github.com/charmbracelet/vhs)
# Output: docs/demo.gif

Output docs/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1100
Set Height 660
Set Theme "Dracula"
Set WindowBar Colorful
Set Padding 10
Set TypingSpeed 60ms
Set WaitTimeout 30s

# ── title ──────────────────────────────────────────────────────────────────
Type "# raggity — local-first RAG answered by Claude"
Sleep 1s
Enter
Sleep 500ms

# ── step 1: ingest ─────────────────────────────────────────────────────────
Type "rag ingest"
Sleep 300ms
Enter
Sleep 8s

# ── step 2: ask a real question that has an answer ─────────────────────────
Type 'rag ask "How does raggity handle selective abstention?"'
Sleep 300ms
Enter
Sleep 12s

# ── step 3: ask something NOT in the knowledge base ────────────────────────
Sleep 500ms
Type 'rag ask "Who won the 2032 FIFA World Cup?"'
Sleep 300ms
Enter
Sleep 8s

Sleep 3s
