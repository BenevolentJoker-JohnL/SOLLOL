// Unified SOLLOL Dashboard Refresh Helper
  (function () {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = location.host;

    function connectOllama() {
      const target = document.getElementById('ollama-activity');
      if (!target) {
        console.error('No #ollama-activity element found');
        return;
      }
      const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/ollama_activity`);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const entry = document.createElement('div');
        entry.style.padding = '0.35rem 0';
        entry.style.borderBottom = '1px solid #1e293b';
        entry.textContent = data.message || event.data;
        target.appendChild(entry);
        target.scrollTop = target.scrollHeight;
      };
      ws.onopen = () => console.log('✅ Ollama activity WebSocket refreshed');
      ws.onerror = (err) => console.error('Ollama activity WS error', err);
      ws.onclose = () => setTimeout(connectOllama, 5000);
    }

    function connectRouting() {
      const target = document.getElementById('routing-events');
      if (!target) {
        console.error('No #routing-events element found');
        return;
      }
      const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/routing_events`);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const entry = document.createElement('div');
        entry.style.padding = '0.35rem 0';
        entry.style.borderBottom = '1px solid #1e293b';
        entry.textContent = data.message || event.data;
        target.appendChild(entry);
        target.scrollTop = target.scrollHeight;
      };
      ws.onopen = () => console.log('✅ Routing events WebSocket refreshed');
      ws.onerror = (err) => console.error('Routing events WS error', err);
      ws.onclose = () => setTimeout(connectRouting, 5000);
    }

    connectOllama();
    connectRouting();
  })();
