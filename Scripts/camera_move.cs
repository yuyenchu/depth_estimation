using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine;

public class camera_move : MonoBehaviour
{
    public Transform target;
    public Texture background;
    public bool save_view = false;
    public int width = 256;
	public int height = 256;
	private int imageCounter = 0;
    private float timer = 0.0f;
    // Start is called before the first frame update
    void Start()
    {
        // this.GetComponent<Camera>().targetTexture = Resources.Load<RenderTexture>(this.name+"_view_depth");
        if(target==null)
            target = GameObject.Find("Center").transform;

        int i = ConvertToInt(this.name);
        if(i!=0)
        {
            GameObject canvas = GameObject.Find("Canvas");

            GameObject bg = new GameObject(this.name+"ui_bg");
            bg.AddComponent<CanvasRenderer>();
            bg.AddComponent<RectTransform>();
            RawImage image =  bg.AddComponent<RawImage>();
            image.texture = background;
            image.color = Color.black;
            bg.transform.position = new Vector3(10+i*130,140,0);
            bg.GetComponent<RectTransform>().sizeDelta = new Vector2(120, 140);
            bg.transform.SetParent(canvas.transform);  

            GameObject view = new GameObject(this.name+"ui_view");
            view.AddComponent<CanvasRenderer>();
            view.AddComponent<RectTransform>();
            RawImage depthView =  view.AddComponent<RawImage>();
            depthView.texture = Resources.Load<RenderTexture>(this.name+"_view_depth");;
            view.transform.position = new Vector3(10+i*130,130,0);
            view.GetComponent<RectTransform>().sizeDelta = new Vector2(100, 100);
            view.transform.SetParent(canvas.transform);  

            GameObject text = new GameObject(this.name+"ui_txt");
            text.AddComponent<CanvasRenderer>();
            Text txt = text.AddComponent<Text>();
            txt.text = this.name+" depth view";
            txt.color = Color.white;
            txt.alignment = TextAnchor.UpperCenter;
            txt.font = Resources.GetBuiltinResource(typeof(Font), "Arial.ttf") as Font;
            text.transform.position = new Vector3(10+i*130,195,0);
            text.GetComponent<RectTransform>().sizeDelta = new Vector2(120, 20);
            text.transform.SetParent(canvas.transform);  
        }

        Debug.Log(this.name+"->"+i);
    }

    // Update is called once per frame
    void Update()
    {
        transform.LookAt(target);
        transform.Translate(Vector3.right * Time.deltaTime);
        timer += Time.deltaTime;
        if(save_view && timer > 1.0) {
            GetComponent<ImageSynthesis>().Save(this.name + "_" + imageCounter++, width, height);
            timer = 0.0f;
        }
    }

    // void obGUI()
    // {
    //     if(background!=null)
    //         GUI.DrawTexture(new Rect(Screen.width - 240,Screen.height-240,240,240), background, ScaleMode.StretchToFill, false, 0, Color.black, 0, 0);
    // }
    int ConvertToInt(String input)
    {
        String inputCleaned = Regex.Replace(input, "[^0-9]", "");
        int value = 0;

        if (int.TryParse(inputCleaned, out value))
            return value;
        return 0;
    }
}
