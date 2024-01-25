$(document).ready(function () {
    
    var classInfo = {
        "Apple Scab": {
            symptoms: "Dark, scaly lesions on leaves, fruit, and twigs of apple trees. Leaves may curl and drop, reducing fruit yield and quality.",
            diseaseCycle: "The fungus overwinters in fallen leaves and fruit on the ground. In spring, as temperatures rise and humidity increases, it produces spores that are carried by wind or rain to new growth. Infection occurs when the spores land on susceptible plant parts.",
            fertilizerSuggestion: "Apply Paras DAP (Diammonium Phosphate) to provide essential phosphorus and nitrogen. Phosphorus helps with root development, while nitrogen promotes overall plant growth and resistance."
        },
        "Apple Blackrot": {
            symptoms: "Circular, brown-black lesions on apple fruit with concentric rings. Affected fruit becomes mummified and shriveled. Leaves may also develop lesions.",
            diseaseCycle: "Overwintering occurs in mummified fruit or cankers on branches. In spring, fungal spores are released during wet weather, infecting new blossoms and fruit. Rain and wind disseminate the spores.",
            fertilizerSuggestion: "Use IPL DAP (Diammonium Phosphate) to boost phosphorus levels, which aids in fruit development and overall tree health."
        },
        "Apple Cedarapple Rust": {
            symptoms: "Bright orange, gelatinous growths (galls) on apple leaves and fruit in early summer. Later, small yellow spots with black dots appear on leaves.",
            diseaseCycle: "This disease requires two hosts: apple and cedar (or juniper). It overwinters on cedar trees as galls. In spring, spores are released, carried to apple trees by wind, and cause infections on apple leaves and fruit.",
            fertilizerSuggestion: "Apply PPL DAP (Diammonium Phosphate) to promote healthy foliage and fruit growth."
        },
        "Apple Healthy": {
            symptoms: "Represents a healthy apple plant.",
            fertilizerSuggestion: "Maintain the tree's overall health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to ensure it receives all essential nutrients."
        },
        "Blueberry Healthy": {
            symptoms: "Represents a healthy blueberry plant.",
            fertilizerSuggestion: "Use GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to support the growth and fruiting of healthy blueberry plants."
        },
        "Cherry Powdery Mildew": {
            symptoms: "White, powdery growth on cherry leaves, fruit, and shoots. Leaves may curl and turn yellow.",
            diseaseCycle: "This fungus thrives in warm, dry conditions. Spores are produced on infected plant surfaces and are dispersed by wind, initiating new infections.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote vigorous growth and strengthen the plant against disease."
        },
        "Cherry Healthy": {
            symptoms: "Represents a healthy cherry plant.",
            fertilizerSuggestion: "Maintain cherry tree health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Corn Cercospora Leaf Spot": {
            symptoms: "Circular, tan to brown lesions with a dark border on corn leaves. Lesions may coalesce, leading to premature leaf death.",
            diseaseCycle: "Overwinters in infected crop debris. Spores are splashed onto new leaves by rain or carried by wind, causing new infections during humid weather.",
            fertilizerSuggestion: "Use GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to support strong corn growth and resilience."
        },
        "Corn Common Rust": {
            symptoms: "Orange to reddish-brown pustules on leaves. Can weaken plants and reduce yield.",
            diseaseCycle: "Spores are produced on infected corn plants. Wind carries them to new plants, and infection occurs in humid conditions, typically in late spring and summer.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to enhance corn plant health and defense mechanisms."
        },
        "Corn Northern Leaf Blight": {
            symptoms: "Long, elliptical lesions with wavy edges on leaves. Can cause significant yield loss.",
            diseaseCycle: "The fungus overwinters in infected crop debris. In spring, spores are carried by wind or rain to new leaves, initiating infections.",
            fertilizerSuggestion: "Use GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote vigorous corn growth and resistance to disease."
        },
        "Corn (Maize) Healthy": {
            symptoms: "Represents a healthy corn plant.",
            fertilizerSuggestion: "Maintain maize plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Grape Black Rot": {
            symptoms: "Circular, dark brown to black lesions on grape leaves and fruit. Affected fruit may become covered in spore-producing structures.",
            diseaseCycle: "The fungus overwinters in infected plant material. In spring, spores are released during wet periods, infecting new growth.",
            fertilizerSuggestion: "Apply SPIC DAP (Diammonium Phosphate) to provide phosphorus and nitrogen, supporting grapevine growth and disease resistance."
        },
        "Grape Esca (Black Measles)": {
            symptoms: "Leaf discoloration, wood necrosis, and dieback in grapevines. Leaves may show red streaks.",
            diseaseCycle: "Esca is a complex disease caused by multiple fungi. Infection often occurs through pruning wounds, and the disease progresses slowly over time.",
            fertilizerSuggestion: "Strengthen the grapevines with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Grape Leaf Blight (Isariopsis Leaf Spot)": {
            symptoms: "Angular, brown lesions with a yellow halo on grape leaves. Severe infections can lead to defoliation.",
            diseaseCycle: "The fungus overwinters in infected leaves on the vine or ground. In spring, spores are spread by rain, wind, or contact, causing new infections.",
            fertilizerSuggestion: "Use SPIC DAP (Diammonium Phosphate) to ensure grapevines receive essential nutrients for growth and disease resistance."
        },
        "Grape Healthy": {
            symptoms: "Represents a healthy grapevine.",
            fertilizerSuggestion: "Maintain grapevine health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Orange Huanglongbing (Citrus Greening)": {
            symptoms: "Yellowing of citrus leaves, misshapen and bitter fruit, dieback of branches, and overall tree decline.",
            diseaseCycle: "Huanglongbing is caused by a bacterium transmitted by the Asian citrus psyllid. Once infected, the bacterium spreads throughout the tree, causing symptoms.",
            fertilizerSuggestion: "Apply a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to support overall tree health and strengthen citrus defenses against disease."
        },
        "Peach Bacterial Spot": {
            symptoms: "Dark, water-soaked lesions on peach leaves and fruit. Lesions may have a purple halo.",
            diseaseCycle: "Bacteria overwinter in cankers on twigs. In spring, they are released during rain and infect new growth through natural openings or wounds.",
            fertilizerSuggestion: "Use PPL DAP (Diammonium Phosphate) to provide phosphorus and nitrogen, supporting peach tree growth and disease resistance."
        },
        "Peach Healthy": {
            symptoms: "Represents a healthy peach tree.",
            fertilizerSuggestion: "Maintain peach tree health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Pepper Bell Bacterial Spot": {
            symptoms: "Dark, raised lesions on pepper leaves and fruit. Lesions may have a white halo.",
            diseaseCycle: "Bacteria survive in infected debris or on seeds. Rain, wind, and contact spread bacteria to new growth, initiating infections.",
            fertilizerSuggestion: "Apply IPL DAP (Diammonium Phosphate) to boost phosphorus levels and enhance pepper plant growth and resistance."
        },
        "Pepper Bell Healthy": {
            symptoms: "Represents a healthy pepper plant.",
            fertilizerSuggestion: "Maintain pepper plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Potato Early Blight": {
            symptoms: "Dark, concentric rings with a target-like appearance on potato leaves. Leaves may yellow and die, reducing yield.",
            diseaseCycle: "Fungal spores overwinter in infected plant debris. Spores are splashed onto new leaves during rain, and infection occurs in warm and humid conditions.",
            fertilizerSuggestion: "Use GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote healthy potato growth and resistance to early blight."
        },
        "Potato Late Blight": {
            symptoms: "Dark, water-soaked lesions on potato leaves and fruit. White, fuzzy growth may appear on the underside of leaves.",
            diseaseCycle: "The fungus spreads through airborne spores during periods of high humidity and rain. Overwintering occurs in infected plant debris.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to strengthen potato plants against late blight and support tuber development."
        },
        "Potato Healthy": {
            symptoms: "Represents a healthy potato plant.",
            fertilizerSuggestion: "Maintain potato plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Raspberry Healthy": {
            symptoms: "Represents a healthy raspberry plant.",
            fertilizerSuggestion: "Maintain raspberry plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Soybean Healthy": {
            symptoms: "Represents a healthy soybean plant.",
            fertilizerSuggestion: "Maintain soybean plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Squash Powdery Mildew": {
            symptoms: "White, powdery coating on squash leaves, often starting on older leaves. Leaves may curl and die prematurely.",
            diseaseCycle: "The fungus thrives in dry conditions with high humidity at night. Spores are carried by wind and germinate on leaf surfaces.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote healthy squash growth and resistance to powdery mildew."
        },
        "Strawberry Leaf Scorch": {
            symptoms: "Reddish discoloration and browning of strawberry leaves, starting at the leaf edges and progressing inward.",
            diseaseCycle: "The bacterium overwinters in infected leaves. Rain splashes the bacterium onto new growth, initiating infections during wet conditions.",
            fertilizerSuggestion: "Use SPIC DAP (Diammonium Phosphate) to ensure strawberry plants receive essential nutrients for growth and disease resistance."
        },
        "Strawberry Healthy": {
            symptoms: "Represents a healthy strawberry plant.",
            fertilizerSuggestion: "Maintain strawberry plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        },
        "Tomato Bacterial Spot": {
            symptoms: "Dark, raised lesions with a yellow halo on tomato leaves and fruit. Lesions may merge under humid conditions.",
            diseaseCycle: "Bacteria overwinter in infected plant debris. Rain and wind disseminate bacteria, initiating infections during wet weather.",
            fertilizerSuggestion: "Apply PPL DAP (Diammonium Phosphate) to provide phosphorus and nitrogen, supporting tomato plant growth and disease resistance."
        },
        "Tomato Early Blight": {
            symptoms: "Dark, concentric rings on tomato leaves, typically starting on lower leaves. Leaves may yellow and die, reducing fruit production.",
            diseaseCycle: "The fungus overwinters in infected plant debris. Spores are splashed onto new leaves during rain, and infection occurs in warm and humid conditions.",
            fertilizerSuggestion: "Use SPIC DAP (Diammonium Phosphate) to promote healthy tomato growth and resistance to early blight."
        },
        "Tomato Late Blight": {
            symptoms: "Dark, water-soaked lesions on tomato leaves and fruit. White, fuzzy growth may appear on the underside of leaves.",
            diseaseCycle: "Spores are windborne and can travel long distances. Infection occurs during cool and wet conditions, particularly in late summer and fall.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to strengthen tomato plants against late blight and support fruit development."
        },
        "Tomato Leaf Mold": {
            symptoms: "Pale, yellowish lesions on tomato leaves, often surrounded by a green halo. Lesions may be confined to the upper leaf surface.",
            diseaseCycle: "Thrives in warm and humid conditions. Spores are produced on the lower leaf surface and are easily disseminated.",
            fertilizerSuggestion: "Use SPIC DAP (Diammonium Phosphate) to ensure tomato plants receive essential nutrients for growth and disease resistance."
        },
        "Tomato Septorial Leaf Spot": {
            symptoms: "Circular, dark lesions with a light center on tomato leaves. Leaves may yellow and drop prematurely.",
            diseaseCycle: "Overwinters in infected plant debris. Spores are splashed onto new leaves by rain, initiating infections during humid weather.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote healthy tomato growth and resistance to septorial leaf spot."
        },
        "Tomato Two-Spotted Spider Mite": {
            symptoms: "Tiny, yellow or white stippling on tomato leaves. Webbing may be present. Leaves may yellow and drop.",
            diseaseCycle: "Spider mites thrive in hot and dry conditions. They feed on plant sap, damaging cells and causing stippling.",
            fertilizerSuggestion: "Use GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to promote healthy tomato growth, as well-tended plants can better withstand mite infestations."
        },
        "Tomato Target Spot": {
            symptoms: "Large, dark concentric rings on tomato leaves. Lesions may have a target-like appearance and a light center.",
            diseaseCycle: "The fungus overwinters in infected plant debris. Spores are carried by wind or rain to new leaves, initiating infections.",
            fertilizerSuggestion: "Apply GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer to strengthen tomato plants against target spot and promote fruit development."
        },
        "Tomato Yellow Leaf Curl Virus": {
            symptoms: "Curling and yellowing of tomato leaves, stunted growth, and reduced fruit yield.",
            diseaseCycle: "Transmitted by whiteflies, the virus infects tomato plants during feeding. Infected whiteflies introduce the virus to new plants.",
            fertilizerSuggestion: "Use IPL DAP (Diammonium Phosphate) to provide phosphorus and nitrogen, which can support tomato plant growth and resistance to viruses."
        },
        "Tomato Mosaic Virus": {
            symptoms: "Mottled or streaked yellow and green leaves, distorted growth, and reduced fruit quality.",
            diseaseCycle: "Spread through contact with infected plant sap or by pests like aphids. The virus can persist in weeds and other hosts.",
            fertilizerSuggestion: "Apply IPL DAP (Diammonium Phosphate) to promote healthy tomato growth and strengthen plants against mosaic viruses."
        },
        "Tomato Healthy": {
            symptoms: "Represents a healthy tomato plant.",
            fertilizerSuggestion: "Maintain tomato plant health with a balanced GENEZIS NPK (Nitrogen Phosphorous Potassium) fertilizer."
        }
    };
    
    
    $("#upload-form").submit(function (e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            type: "POST",
            url: "/predict",
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                if (response.error) {
                    $("#prediction-result").text("Error: " + response.error);
                } else {
                    var predictedClass = response.prediction;
                    $("#prediction-result").text("Prediction: " + predictedClass);

                    // Display class information in the right column
                    var info = classInfo[predictedClass];
                    if (info) {
                        var infoHtml = "<h3>Symptoms:</h3><p>" + info.symptoms + "</p>";
                        infoHtml += "<h3>Disease Cycle:</h3><p>" + info.diseaseCycle + "</p>";
                        infoHtml += "<h3>Fertilizer:</h3><p>" + info.fertilizerSuggestion + "</p>";
                        $(".right-column").html(infoHtml);
                    } else {
                        $(".right-column").html(""); // Clear right column if no info is available
                    }
                }
            },
            error: function () {
                $("#prediction-result").text("An error occurred.");
            }
        });
    });


    const fileInput = document.getElementById("file-input");
    const imagePreview = document.getElementById("image-preview");

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                // Set the source of the image preview to the selected file's data URL
                imagePreview.src = e.target.result;
            };

            reader.readAsDataURL(file);
        } else {
            // If no file is selected or the selection is canceled, clear the image preview
            imagePreview.src = "upload-icon.png"; // You can replace with your default icon/image
        }
    });





});