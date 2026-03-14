import { data } from "react-router-dom";

export default function ColourPalette() {

    if (!data || data.length === 0) return null

    return(
        <div style={{ marginTop: "20px" }}>
            <h3>Colour Palette</h3>
            <div style={{ display: "flex", gap: "10px"}}>
                {data.map((item) => (
                    <div key={item.colour} style={{ textAlign: "center" }}>

                        <div style={{
                            width: "60px",
                            height: "60px",
                            background: item.colour.toLowerCase(),
                            border: "1px solid #ccc"}}/>
                        
                        <small>{item.colour}</small>
                    </div>
                ))}
            </div>
        </div>
    )
}