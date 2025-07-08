import React from 'react';
import './signup.module.css'; // Import the CSS file

const Signup: React.FC = () => {
    return (
        <div className="signup-container">
            {/* Logo section */}
            <div className="logo">
                <span className="logo-icon">🤖</span> Samantha
            </div>

            {/* Form content */}
            <div className="form-content">
                <h1>Create Account</h1>
                <p className="subheading">Join the next generation of innovation</p>
                
                <div className="form-group">
                    <label className="input-label">Full Name</label>
                    <input type="text" placeholder="Enter your name" />
                </div>
                
                <div className="form-group">
                    <label className="input-label">Email</label>
                    <input type="email" placeholder="you@example.com" />
                </div>
                
                <div className="form-group">
                    <label className="input-label">Password</label>
                    <input type="password" placeholder="••••••••" />
                </div>
                
                <button>Create Account</button>
                
                <p className="login-link">Already have an account? <a href="/" className="login-link">Log in</a></p>
            </div>
        </div>
    );
};

export default Signup;