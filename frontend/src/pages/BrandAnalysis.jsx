import { useEffect, useState } from "react"
import {getBrandColours} from "../services/api"
import ColourCharts from "../components/colourcharts"
import BrandSelector from "../components/BrandSelector"

export default function BrandAnalysis() {

const brands = [
  "adage_studio_project_x_unrefyned",
  "adama_paris",
  "ajabeng",
  "ajanee",
  "babayo",
  "boyedoe",
  "cynthia_abila",
  "desiree_iyama",
  "dimeji_ilori",
  "eki_silk",
  "elexiay",
  "emmy_kasbit",
  "eso_by_liman",
  "for_style_sake",
  "fruche",
  "hawa_paris",
  "hertunba",
  "ibilola_ogundipe",
  "imad_eduso",
  "jzo",
  "last_three",
  "lb_lumina",
  "left_of_yaba_x_jilk",
  "lfj",
  "lila_bare",
  "maison_alulla",
  "maxjenny",
  "mot_the_label",
  "ndiiche_x_sinae",
  "nkwo",
  "nya",
  "olooh",
  "oshobor",
  "pepperrow",
  "pettre_taylor",
  "rendoll",
  "revival_london",
  "sahrazad",
  "sevon_dejana",
  "street_souk",
  "studio_imo",
  "the_or_foundation",
  "wote",
  "y'wande"]
    const [data, setData] = useState([])
    const [brand, setBrand] = useState([0])
    const [loading, setloading] = useState(true)

    useEffect(() => {
        getBrandColours(brand)
        .then(response => {
            console.log("Fetched Brand Colours:", response); // log the fetched data for debugging
            setData(response); // store the response data in state
            setloading(false);
        })

    .catch((error) =>{
        console.error("Error fetching brand colours:", error);
            setloading(false);
    })
    }, [brand])

    if (loading) return <p>Loading...</p>
    // !loading && <p>No colours found here</p>
    console.log("Current Data State:", data);
    
    return (
        <div style={{ padding: "20px" }}>
            <h1>LAGOS FASHION WEEK F/W 2025 COLOUR ANALYSIS</h1>

            <BrandSelector brands={brands} 
            onSelect={setBrand} />

            {!loading && data?.brand_colours?.length > 0 ?(
            <ColourCharts data={data.brand_colours} />
            ) : (
            <p>Loading chart data...</p>
            )}
        </div>
    )
};
