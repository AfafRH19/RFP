import "./App.css";
import MainForm from "./Component/MainForm";
import "./Component/NavBar";
import Navbar from "./Component/NavBar";
import "./styles.css";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Navbar />
      </header>
     <MainForm />
    </div>
  );
}

export default App;
