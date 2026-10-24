const { createApp } = Vue;
createApp({
  delimiters: ['[[', ']]'],
  data: () => ({
    tab: 'predict', mode: 'wheat',
    modes: [
      { id: 'wheat', name: '小麦麦穗分割', desc: 'RGB图像像素级分割' },
      { id: 'growth', name: '大麦长势分类', desc: '正常旺盛或干枯衰退' },
      { id: 'lodging', name: '大麦倒伏区域', desc: '图块级倒伏区域识别' }
    ],
    files: [], results: [], history: [], health: {}, loading: false, batchDownloadUrl: '',
    dragging: false, error: '', batchElapsed: 0
  }),
  computed: { modeName() { return this.modes.find(item => item.id === this.mode).name; } },
  mounted() { this.checkHealth(); },
  methods: {
    async checkHealth() {
      try { this.health = await (await fetch('/api/health')).json(); }
      catch { this.health = { status: 'error' }; }
    },
    selectFiles(event) { this.addFiles([...event.target.files]); event.target.value = ''; },
    onDrop(event) { this.dragging = false; this.addFiles([...event.dataTransfer.files]); },
    addFiles(items) {
      this.error = '';
      const valid = items.filter(file => (this.mode === 'wheat' ? /\.(png|jpe?g|tiff?)$/i : /\.tiff?$/i).test(file.name));
      this.files = [...this.files, ...valid].slice(0, 10);
      if (valid.length !== items.length) this.error = '已忽略不符合当前模式的文件';
    },
    removeFile(index) { this.files.splice(index, 1); }, clearFiles() { this.files = []; },
    async predict() {
      if (!this.files.length) return;
      this.loading = true; this.error = ''; this.batchDownloadUrl = '';
      const data = new FormData(); data.append('mode', this.mode);
      this.files.forEach(file => data.append('images', file));
      try {
        const response = await fetch('/api/predict', { method: 'POST', body: data });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || '分析失败');
        this.results = body.results; this.batchElapsed = body.batch_elapsed;
        this.batchDownloadUrl = body.batch_download_url || ''; this.files = [];
      } catch (error) { this.error = error.message; } finally { this.loading = false; }
    },
    async loadHistory() {
      try { const response = await fetch('/api/history'); this.history = (await response.json()).items || []; }
      catch { this.error = '历史记录加载失败'; }
    },
    async removeHistory(id) {
      if (!confirm('确定删除这条记录及输出文件吗？')) return;
      const response = await fetch(`/api/history/${id}`, { method: 'DELETE' });
      if (response.ok) this.history = this.history.filter(item => item.run_id !== id);
    },
    percent(value) { return `${(Number(value) * 100).toFixed(2)}%`; },
    formatBytes(value) { return value < 1048576 ? `${(value / 1024).toFixed(1)}KB` : `${(value / 1048576).toFixed(1)}MB`; },
    formatTime(value) { return new Date(value).toLocaleString('zh-CN'); }
  }
}).mount('#app');
