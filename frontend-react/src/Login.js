import { useState } from "react";

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function submit(type) {
    const res = await fetch(`${process.env.REACT_APP_API_URL}/api/${type}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });

    const data = await res.json();
    if (data.token) {
      localStorage.setItem("token", data.token);
      onLogin();
    } else {
      alert(data.error);
    }
  }

  return (
    <div style={{ padding: 40 }}>
      <h2>Login</h2>
      <input placeholder="Email" onChange={e => setEmail(e.target.value)} /><br/>
      <input type="password" placeholder="Password" onChange={e => setPassword(e.target.value)} /><br/>
      <button onClick={() => submit("login")}>Login</button>
      <button onClick={() => submit("register")}>Register</button>
    </div>
  );
}
