import { useState } from "react";

export default function Login({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function submit() {
    const endpoint = isRegister ? "register" : "login";

    const body = isRegister
      ? { username, email, password }
      : { username, password };

    const res = await fetch(`${process.env.REACT_APP_API_URL}/api/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const data = await res.json();
    if (data.token) {
      localStorage.setItem("token", data.token);
      onLogin();
    } else {
      alert(data.error || "Authentication failed");
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.title}>{isRegister ? "Create Account" : "Welcome Back"}</h2>
        <p style={styles.subtitle}>
          {isRegister ? "Register to continue" : "Login to continue"}
        </p>

        <input
          style={styles.input}
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />

        {isRegister && (
          <input
            style={styles.input}
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
        )}

        <input
          style={styles.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />

        <button style={styles.primaryBtn} onClick={submit}>
          {isRegister ? "Register" : "Login"}
        </button>

        <button style={styles.linkBtn} onClick={() => setIsRegister(!isRegister)}>
          {isRegister ? "Already have an account? Login" : "New here? Create account"}
        </button>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #4f46e5, #6366f1)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center"
  },
  card: {
    background: "white",
    padding: "40px",
    borderRadius: "12px",
    width: "100%",
    maxWidth: "360px",
    boxShadow: "0 20px 40px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center"
  },
  title: {
    marginBottom: "4px"
  },
  subtitle: {
    marginBottom: "24px",
    fontSize: "14px",
    color: "#555"
  },
  input: {
    width: "100%",
    padding: "12px",
    marginBottom: "14px",
    borderRadius: "6px",
    border: "1px solid #ddd",
    fontSize: "14px"
  },
  primaryBtn: {
    width: "100%",
    padding: "12px",
    background: "#4f46e5",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "15px",
    marginTop: "6px"
  },
  linkBtn: {
    marginTop: "12px",
    background: "none",
    border: "none",
    color: "#4f46e5",
    cursor: "pointer",
    fontSize: "13px"
  }
};
