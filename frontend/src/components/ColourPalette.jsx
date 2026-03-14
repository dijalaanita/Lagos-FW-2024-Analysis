export default function ColourPalette({data}) {

    if (!data || data.length === 0) return null

    return(
        <div style={{ marginTop: "20px" }}>
            <h3>Colour Palette</h3>
            <div style={{ display: "flex", gap: "10px"}}>
                {data.map((item) => {
                    const displayColour = item.colour.split('/')[0].trim().toLowerCase(); // Extract the main colour name and convert to lowercase
    
                    <div key={item.colour} style={{ textAlign: "center" }}>

                        <div style={{
                            width: "60px",
                            height: "60px",
                            background: item.colour.toLowerCase(),
                            border: "1px solid #ccc"}}/>
                        
                        <small style={{ display: "block", marginTop: "5px" }}>
                            {item.colour}</small>
                    </div>
})}
            </div>
        </div>
    );
}