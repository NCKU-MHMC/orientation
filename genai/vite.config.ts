// 文獻連結一律開新分頁,簡報頁面不被帶走。
// 不從 'vite' import defineConfig:vite 只是 @slidev/cli 的傳遞相依,
// CI 用 npm ci --omit=dev 時直接 import 會多一個失敗點。
export default ({
  slidev: {
    markdown: {
      markdownItSetup(md) {
        const dflt = md.renderer.rules.link_open
          || ((tokens, idx, opts, _env, self) => self.renderToken(tokens, idx, opts))
        md.renderer.rules.link_open = (tokens, idx, opts, env, self) => {
          if (/^https?:/.test(tokens[idx].attrGet('href') || '')) {
            tokens[idx].attrSet('target', '_blank')
            tokens[idx].attrSet('rel', 'noopener')
          }
          return dflt(tokens, idx, opts, env, self)
        }
      },
    },
  },
})
