import { useEffect, useState } from "react"
import {getBrandColours} from "../services/api"
//import ColourCharts from "../components/colourcharts"

export default function BrandAnalysis() {
    const [data, setData] = useState([])
    const [loading, setloading] = useState(true)

    useEffect(() => {
        getBrandColours("Babayo")
        .then(response => {
            console.log("Fetched Brand Colours:", response); // log the fetched data for debugging
            setData(response); // store the response data in state
            setloading(false);
        })

    .catch((error) =>{
        console.error("Error fetching brand colours:", error);
            setloading(false);
    })
    }, [])

    if (loading) return <p>Loading...</p>
    // !loading && <p>No colours found here</p>
    console.log("Current Data State:", data);
    
    return (
        <div style={{ padding: "20px" }}>
            <h1>Babayo Colour Analysis</h1>
            
            <div style={{display: 'flex', gap:'10px', flexWrap: 'wrap'}}>
                {data && (data.brand_colours || data).map((item,index) => {
                    const hexColour = item.colours?.startsWith('#') ? item.hex : `#${item.colours}`;
                    if (!item.hex) console.warn(`Missing hex value for item at index ${index}:`, item);
                    
                    return (
                        <div key={index} style={{ textAlign: 'center'}}>
                            <div style={{
                            backgroundColor: item.colours ? hexColour : '#ccc', // fallback color if hex is missing 
                            width: '80px',
                            height: '80px',
                            borderRadius: '12px',
                            border: '1px solid #ab3333', // add a border to make the circles more visible against the background
                            boxShadow: '0 2px 5px rgba(0, 0, 0, 0.1)' // add a subtle shadow for better visibility
                            //alignItems: 'center',
                            //justifyContent: 'center',
                            //display: 'flex',
                            //color: '#fff',
                            //fontSize: '10px',
                            //textShadow: '1px 1px 2px #000' // add text shadow for better readability
                }} />
                <p style={{marginTop: '5px', fontSize: '12px', color: '#333', fontWeight: 'bold'}}>
                {item.colours || 'No Hex'}
                </p>
                </div>
                );
                })}
            </div>
                    <p style={{ marginTop:'20px', color: '#888' }}>
                        Showing {(data.brand_colours || data).length} colours from the runway.
                    </p>

        </div>

    );
};