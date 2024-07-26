import React from "react";
import "./NavBar.css";
import logo from "../media/logo.png";

const Navbar = () => {
  return (
    <header>
      <nav className="navbar">
        <div className="logo">
          <img src={logo} alt="Harmonic Logo" />
        </div>
        <ul className="nav-links">
          <li>
            <a href="#home">Home</a>
          </li>
          <li>
            <a href="#about">About</a>
          </li>
          <li>
            <a href="#contact">Contact</a>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Navbar;
