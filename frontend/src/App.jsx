import BrandAnalysis from "./pages/BrandAnalysis"
import NavBar from "./components/NavBar"
import Overview from "./pages/Overview"

export default function App() {
  return (
    <div>
      <NavBar />
    
    <div style={{ 
  padding: "40px", 
  // maxWidth: "1200px", 
  // margin: "0 auto", 
  // fontFamily: "'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  // backgroundColor: "#f9f9f9",
  // minHeight: "100vh"
}}>
  <Routes>
    <Routes path="/" element={<Overview />} />

    <Routes path="/brand-analysis" element={<BrandAnalysis />} />
  </Routes>
  {/* <Overview />
  <BrandAnalysis /> */}
</div>
</div>
  )
}