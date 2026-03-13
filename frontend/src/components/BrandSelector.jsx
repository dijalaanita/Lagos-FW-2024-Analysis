export default function BrandSelector({ brands, onSelect }) {

    return (
        <div style={{ marginBottom: "20px" }}>
            <label>Select Brand:</label>
            <select onChange={(e) => onSelect(e.target.value)} 
            style={{ marginLeft: "10px", padding: "5px" }}>

                {brands.map((brand) => (
                    <option key={brand} value={brand}>{brand}</option>
                ))}
            </select>
        </div>
    )

}