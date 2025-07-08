import React from 'react';
import './login.module.css'; // Import the CSS file

const Login: React.FC = () => {
    return (
        <div className="login-container">
            <div className="login-content">
                <h1>Hi, I am Samatha</h1>
                <p className="login-subtitle">Sign in to your premium AI assistant</p>
                
                <div className="form-group">
                    <input type="email" placeholder="Email address" />
                </div>
                
                <div className="form-group">
                    <input type="password" placeholder="Password" />
                </div>
                
                <div className="remember-me">
                    <input type="checkbox" id="remember" />
                    <label htmlFor="remember">Remember me for 30 days</label>
                </div>
                
                <button className="login-button">Log In</button>
                
                <div className="login-links">
                    <a href="/forgot-password" className="forgot-link">Forgot password?</a>
                    <p className="signup-text">Don't have an account? <a href="/signup" className="signup-link">Sign up</a></p>
                </div>
            </div>
        </div>
    );
};

export default Login;