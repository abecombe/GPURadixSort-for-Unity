using UnityEngine;

public class FPSCounter : MonoBehaviour
{
    [SerializeField]
    private int _fontSize = 32;
    [SerializeField]
    private Color _color = Color.white;

    private float _deltaTime = 0.0f;

    private void Update()
    {
        _deltaTime += (Time.deltaTime - _deltaTime) * 0.1f;
    }

    private void OnGUI()
    {
        float msec = _deltaTime / Time.timeScale * 1000.0f;
        float fps = Time.timeScale / _deltaTime;

        string text = string.Format("{0:0.00} ms ({1:0.00} fps)", msec, fps);

        Rect rect = new()
        {
            x = Screen.width
        };

        GUIStyle style = new()
        {
            alignment = TextAnchor.UpperRight,
            fontSize = _fontSize
        };
        style.normal.textColor = _color;

        GUI.Label(rect, text, style);
    }
}