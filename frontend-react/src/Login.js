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
    <div style={{ padding: 40 }}>
      <h2>{isRegister ? "Register" : "Login"}</h2>

      {isRegister && (
        <>
          <input
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
          /><br/>
          <input
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
          /><br/>
        </>
      )}

      {!isRegister && (
        <input
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
      )}

      <br/>

      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
      /><br/>

      <button onClick={submit}>
        {isRegister ? "Register" : "Login"}
      </button>

      <button
        onClick={() => setIsRegister(!isRegister)}
        style={{ marginLeft: 10 }}
      >
        {isRegister ? "Go to Login" : "Create Account"}
      </button>
    </div>
  );
}
