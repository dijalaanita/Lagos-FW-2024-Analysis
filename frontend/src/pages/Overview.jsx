import { useEffect, useState } from "react"
import { getOverviewColours } from "../services/api"
import ColourCharts from "../components/colourcharts"
import Insights from "../components/Insights"

export default function Overview(){
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {

        getOverviewColours()
        .then((response) => {
            setData(response.data);
            setLoading(false);})
        .catch((error) => {
            console.error("failed to fetch colours", error);
            setLoading(false);
        });

    }, []);

    if (loading) return <div>loading overall fashion trends...</div>
    console.log("Overview Data:", data);
    return (
        <div>
            <h1>LAGOS FASHION WEEK COLOUR TRENDS</h1>
            <ColourCharts data={data}/>
            <Insights data={data}/>
            </div>
    )
}