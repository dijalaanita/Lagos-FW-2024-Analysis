import axios from "axios"

const API_BASE = axios.create({
  baseURL: "http://localhost:8000",
})

export const getBrandColours = async (brand) => {
    const response = await fetch(`http://127.0.0.1:8000/colours/brand/${brand}`)
    if (!response.ok) throw new Error(`Failed to fetch brand colours for ${brand}: ${response.statusText}`)
    return await response.json();
};