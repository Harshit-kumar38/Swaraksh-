import { useEffect, useRef, useState } from "react";
import "./index.css";

const WS_URL = "ws://127.0.0.1:8000/stream";

function App() {
  const socketRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("WAITING");
  const [spoofScore, setSpoofScore] = useState(0);
  const [bonaFideScore, setBonaFideScore] = useState(1);
  const [latency, setLatency] = useState(0);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [callSid, setCallSid] = useState("-");
  const [history, setHistory] = useState([]);

  const connectWebSocket = () => {
    if (
      socketRef.current &&
      socketRef.current.readyState === WebSocket.OPEN
    ) {
      return;
    }

    const socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      console.log("WebSocket connected");

      setConnected(true);
      setStatus("CONNECTED");

      socket.send(
        JSON.stringify({
          event: "start",
          start: {
            callSid: "FRONTEND_TEST_001",
          },
        })
      );
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      console.log("Server:", data);

      if (data.event === "connected") {
        setStatus("MONITORING");
      }

      if (data.event === "analysis") {
        const spoof = Number(data.spoof_score);

        setSpoofScore(spoof);
        setBonaFideScore(Number(data.bona_fide_score));
        setLatency(Number(data.latency_ms || 0));
        setAnalysisCount(Number(data.analysis_number || 0));

        if (data.callSid) {
          setCallSid(data.callSid);
        }

        setStatus(
          data.is_spoof
            ? "DEEPFAKE DETECTED"
            : "LIKELY REAL"
        );

        setHistory((previous) => [
          {
            number: data.analysis_number,
            score: spoof,
            result: data.result,
            latency: data.latency_ms,
          },
          ...previous,
        ].slice(0, 10));
      }

      if (data.event === "final_analysis") {
        const spoof = Number(data.spoof_score);

        setSpoofScore(spoof);
        setBonaFideScore(Number(data.bona_fide_score));

        if (data.callSid) {
          setCallSid(data.callSid);
        }

        setStatus(
          data.is_spoof
            ? "DEEPFAKE DETECTED"
            : "LIKELY REAL"
        );

        setHistory((previous) => [
          {
            number: "FINAL",
            score: spoof,
            result: data.result,
            latency: "-",
          },
          ...previous,
        ].slice(0, 10));
      }

      if (data.event === "spoof_detected") {
        setStatus("DEEPFAKE DETECTED");
        setSpoofScore(Number(data.score));
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("CONNECTION ERROR");
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected");

      setConnected(false);

      if (status !== "DEEPFAKE DETECTED") {
        setStatus("DISCONNECTED");
      }
    };

    socketRef.current = socket;
  };

  const disconnectWebSocket = () => {
    if (socketRef.current) {
      socketRef.current.send(
        JSON.stringify({
          event: "stop",
        })
      );

      socketRef.current.close();
      socketRef.current = null;
    }

    setConnected(false);
    setStatus("DISCONNECTED");
  };

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const spoofPercentage = (spoofScore * 100).toFixed(2);
  const realPercentage = (bonaFideScore * 100).toFixed(2);

  const isDeepfake = spoofScore >= 0.85;

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">

        <div>
          <div className="brand">
            VOICE<span>GUARD</span>
          </div>

          <div className="subtitle">
            AI VOICE DEEPFAKE DETECTION SOC
          </div>
        </div>

        <div className="system-status">

          <span
            className={
              connected
                ? "status-dot online"
                : "status-dot"
            }
          />

          SYSTEM {connected ? "ONLINE" : "OFFLINE"}

        </div>

      </header>


      {/* MAIN */}

      <main className="dashboard">

        {/* TOP CARDS */}

        <section className="grid">

          <div className="card">

            <div className="card-title">
              CONNECTION
            </div>

            <div className="big-value">
              {connected ? "CONNECTED" : "OFFLINE"}
            </div>

            <div className="small">
              WebSocket
            </div>

          </div>


          <div className="card">

            <div className="card-title">
              MODEL
            </div>

            <div className="big-value">
              AASIST-L
            </div>

            <div className="small">
              Apple CoreML
            </div>

          </div>


          <div className="card">

            <div className="card-title">
              ANALYSES
            </div>

            <div className="big-value">
              {analysisCount}
            </div>

            <div className="small">
              Analysis windows
            </div>

          </div>


          <div className="card">

            <div className="card-title">
              LATENCY
            </div>

            <div className="big-value">
              {latency ? `${latency.toFixed(0)} ms` : "--"}
            </div>

            <div className="small">
              Inference
            </div>

          </div>

        </section>


        {/* DETECTION PANEL */}

        <section className="detection-card">

          <div className="card-title">
            LIVE VOICE ANALYSIS
          </div>

          <div
            className={
              isDeepfake
                ? "detection-status danger"
                : "detection-status safe"
            }
          >

            {status}

          </div>


          <div className="score-container">

            <div className="score">

              {spoofPercentage}

              <span>%</span>

            </div>

            <div className="score-label">
              SPOOF PROBABILITY
            </div>

          </div>


          <div className="progress">

            <div
              className={
                isDeepfake
                  ? "progress-fill danger-fill"
                  : "progress-fill"
              }
              style={{
                width: `${Math.min(
                  spoofScore * 100,
                  100
                )}%`,
              }}
            />

          </div>


          <div className="score-row">

            <div>
              <span>LIKELY REAL</span>
              <strong>{realPercentage}%</strong>
            </div>

            <div>
              <span>SPOOF</span>
              <strong>{spoofPercentage}%</strong>
            </div>

          </div>

        </section>


        {/* CONTROL */}

        <section className="control-card">

          <div>

            <div className="card-title">
              STREAM CONTROL
            </div>

            <div className="call-id">
              CALL: {callSid}
            </div>

          </div>


          {!connected ? (

            <button
              className="connect-button"
              onClick={connectWebSocket}
            >
              CONNECT DETECTOR
            </button>

          ) : (

            <button
              className="disconnect-button"
              onClick={disconnectWebSocket}
            >
              STOP STREAM
            </button>

          )}

        </section>


        {/* HISTORY */}

        <section className="history-card">

          <div className="card-title">
            ANALYSIS HISTORY
          </div>

          {history.length === 0 ? (

            <div className="empty">
              No analysis results yet.
            </div>

          ) : (

            <div className="history-list">

              {history.map((item, index) => (

                <div
                  className="history-row"
                  key={index}
                >

                  <span>
                    #{item.number}
                  </span>

                  <strong>
                    {(item.score * 100).toFixed(2)}%
                  </strong>

                  <span
                    className={
                      item.score >= 0.85
                        ? "danger-text"
                        : "safe-text"
                    }
                  >
                    {item.result}
                  </span>

                  <span>
                    {item.latency === "-"
                      ? "-"
                      : `${Number(item.latency).toFixed(0)} ms`}
                  </span>

                </div>

              ))}

            </div>

          )}

        </section>

      </main>

    </div>
  );
}

export default App;